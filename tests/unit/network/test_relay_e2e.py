"""The relay end to end: two "devices" on one host, over real TCP.

This is the test that makes the unit invariants mean something: two config roots,
two identities, two relays, real sockets, and the whole protocol driven the way the
CLI drives it. It covers R1 (a relay per install that owns no session), R3 (the
human-confirmed pairing), R4 (zero trust: an unauthorised frame is refused) and R5
(revocation without visiting the revoked device).
"""

from __future__ import annotations

import ast
import threading
import time
from argparse import Namespace
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest

from local_operator.agents import AgentEditFields, AgentRegistry
from local_operator.network import audit as audit_mod
from local_operator.network import cli as net_cli
from local_operator.network import identity
from local_operator.network import invite as invite_mod
from local_operator.network import relay, store, types, wire
from local_operator.network.handshake import (
    Credential,
    Handshake,
    pair_abort_frame,
    pair_timeout_seconds,
    sas_matches,
)
from local_operator.paths import config_dir as ambient_config_dir
from local_operator.session.retention import SESSIONS_DIRNAME
from tests.unit.network import conftest as net_fixtures

NETWORK_NAME = "home-net"


def serve_shaped_relay(
    root: Path, monkeypatch: pytest.MonkeyPatch, **overrides: Any
) -> relay.RelayServer:
    """Build a relay THE WAY ``lop network serve`` BUILDS ONE — with no ``root=``.

    WHY THIS IS THE DEFAULT HERE AND NOT A CONVENIENCE. Both production construction
    sites (``network/cli.py``: the serve command and the tool's engage path) pass
    settings and an identity only, so the config dir is resolved from the AMBIENT
    environment (``paths.config_dir()``). ``RelayServer.__init__`` turns that into
    ``self.root``, and everything path-shaped hangs off it — including the authoriser's
    ``StoreView``. A cell that supplies an explicit root therefore exercises a keyword
    the product never supplies: 25 of 26 constructions in this package did exactly that,
    and the one relational question they never asked of a serve-shaped relay
    (``StoreView.replica_owner``) was answered ``""`` on the real path, which refused
    every automatic replica sync across a real host boundary while a manual
    ``lop sessions sync`` worked (cross-host QA, Q-XH-1; agent review round 1, MAJOR 3).
    Build through here so a raw-root regression cannot hide behind that coincidence
    WHERE THIS HELPER REACHES — the shared ``devices`` fixture and its eleven consumers,
    not every construction in the package: 17 ``RelayServer(…)`` calls in
    ``tests/unit/network`` still pass an explicit root, recorded on the PR as deferred
    (agent review round 2, MINOR 4). The scope is stated because this sentence claimed
    the package on its first draft, and an over-claimed guard is what the round-1
    finding was about.

    ``LOCAL_OPERATOR_CONFIG_DIR`` is what makes "no root argument" mean THIS test's own
    tree rather than the operator's live install, so the assignment is part of the
    construction rather than something a caller might forget.

    THE ASSERTION COMPARES AGAINST THE AMBIENT RESOLUTION, not against the value it was
    handed. ``server.root == root`` read as a guard on the environment but was true by
    identity for the edit it was being credited with: a body that added
    ``kwargs["root"] = root`` would have satisfied it silently (agent review round 2,
    NIT 4). A caller passing ``root=`` is refused by the signature outright
    (``TypeError: got multiple values for argument 'root'``), which is the stronger
    guard; what is left to catch is the environment not taking, and that is exactly what
    the ambient resolution — ``paths.config_dir()``, the same call ``RelayServer``
    makes — answers.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    kwargs: dict[str, Any] = {"settings": relay.NetworkSettings(port=0, listen_address="127.0.0.1")}
    kwargs.update(overrides)
    server = relay.RelayServer(**kwargs)
    assert server.root == ambient_config_dir(), (
        f"a serve-shaped relay resolved {server.root} rather than the ambient "
        f"{ambient_config_dir()}: the test env is not what the product would see"
    )
    assert server.root == root, f"the ambient config dir is not this test's root: {root}"
    return server


@pytest.fixture()
def devices(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[relay.RelayServer, relay.RelayServer, str, int]]:
    """Two relays on loopback: A listening, B able to dial it."""
    root_a = root / "a"
    root_b = root / "b"
    identity_a = identity.mint(root_a, name="device-a")
    identity_b = identity.mint(root_b, name="device-b")
    # Serve-shaped, both of them: this fixture's relays back the pairing, pilot and
    # listing matrix, so the product's own construction is what those cells exercise.
    server_a = serve_shaped_relay(
        root_a, monkeypatch, identity=identity_a, audit=audit_mod.AuditLog(root_a)
    )
    server_b = serve_shaped_relay(
        root_b, monkeypatch, identity=identity_b, audit=audit_mod.AuditLog(root_b)
    )
    host, port = server_a.bind()
    server_a.bind_control()
    server_a.start()
    # The CLI's joining half resolves the config dir from the AMBIENT environment
    # (that is how a user runs it), so the test points the ambient dir at B's root:
    # otherwise the join would write into the isolated HOME and the test would be
    # asserting about a file the code never touched.
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root_b))
    try:
        yield server_a, server_b, host, port
    finally:
        server_a.stop()
        server_b.stop()


def _init_network(server: relay.RelayServer, *, role: str = "admin") -> types.NetworkRecord:
    """Create a network on A and persist it, exactly as `lop network init` does."""
    record = types.NetworkRecord(
        network_id=store.new_network_id(),
        name=NETWORK_NAME,
        created_by=server.identity.device_id,
        self_device_id=server.identity.device_id,
        self_role=role,
        self_capabilities=sorted(types.capabilities_for_role(role)),
        listen={"address": "127.0.0.1", "port": server.settings.port, "advertised": []},
    )
    from secrets import token_bytes

    state = types.SecretState(
        network_id=record.network_id, epoch=1, secret=wire.b64u(token_bytes(32))
    )
    relay.admit(
        record,
        device_id=server.identity.device_id,
        public_key=server.identity.public_key,
        name=server.identity.name,
        role=role,
        capabilities=sorted(types.capabilities_for_role(role)),
        added_by=server.identity.device_id,
        added_via="self",
        root=server.root,
        persist=False,
    )
    store.save(record, server.root)
    store.save_secrets(state, server.root)
    return record


def _mint_invite(server: relay.RelayServer, record: types.NetworkRecord) -> tuple[str, Any]:
    state = store.load_secrets(record.network_id, server.root)
    minted = invite_mod.mint(record, state.secret, role="drive", ttl_s=600.0)
    record.invites.append(minted.record)
    store.save(record, server.root)
    path = store.save_invite_token(minted.record.invite_id, minted.token, server.root)
    assert path.exists()
    return minted.token, minted.envelope


def _join(
    server_b: relay.RelayServer,
    *,
    host: str,
    port: int,
    token: str,
    envelope: Any,
    typed_code: str = "000000",
    settings: relay.NetworkSettings | None = None,
) -> Any:
    """Drive the joining side's ceremony, with the human step supplied by the test.

    The human on the INVITER side is stubbed in the tests below (the relay asks a
    person whether their screen shows the transcribed code); the human on THIS side
    is the ``typed_code`` argument. Both halves of the ceremony are therefore real
    frames over a real socket, and only the two people are simulated — which is what
    a machine is allowed to simulate and nothing more.
    """
    args = Namespace(
        sas_stdin=True,
        verify=False,
        emit_sas=True,
        name=server_b.identity.name,
        json=True,
    )
    return net_cli._join_one(  # noqa: SLF001 — the CLI's own driver, exercised as the CLI runs it
        host=f"{host}:{port}",
        token=token,
        envelope=envelope,
        identity=server_b.identity,
        settings=settings or relay.NetworkSettings(port=0, listen_address="127.0.0.1"),
        args=args,
        wire=wire,
        Handshake=Handshake,
        Credential=Credential,
        pair_abort_frame=pair_abort_frame,
        pair_timeout_seconds=pair_timeout_seconds,
        sas_matches=sas_matches,
        invite_mod=invite_mod,
        store=store,
        relay_mod=relay,
    )


def _type_the_code(monkeypatch: pytest.MonkeyPatch, code: str | None = None) -> None:
    """The JOINER's person. ``None`` means "read the right digits off the other
    screen" — `_read_code` is handed the derived value, so this models a correct
    human rather than bypassing the check. A literal ``code`` models a wrong one."""

    def fake_read_code(args: Any, derived: str, fingerprint: str) -> str:
        del args, fingerprint
        return derived if code is None else code

    monkeypatch.setattr(net_cli, "_read_code", fake_read_code)


def _answer_confirmation(
    server: relay.RelayServer, *, admit: bool = True, timeout: float = 20.0
) -> dict[str, Any] | None:
    """The INVITER's person, standing at `lop network confirm`.

    The relay has no terminal here (a daemon, and a test process besides), so it
    parks the pairing in a 0600 pending record carrying BOTH codes; this waits for
    that record and answers it through the relay's own control op — the same path
    the CLI takes, audit record included. Returns the parked row it answered, or
    ``None`` when none appeared, which is what the refusal cases assert.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        rows = server._ctl_pair_pending({})  # noqa: SLF001 — the CLI's own control op
        if rows:
            server._ctl_pair_confirm(  # noqa: SLF001
                {
                    "invite_id": rows[0]["invite_id"],
                    "decision": "admit" if admit else "decline",
                    "matched": admit,
                    "reason": "" if admit else "declined",
                    "answered_by": "harness",
                }
            )
            return dict(rows[0])
        time.sleep(0.05)
    return None


# ---------------------------------------------------------------------------
# R3 — pairing
# ---------------------------------------------------------------------------


def test_pairing_admits_the_joiner_after_both_people_confirm(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole ceremony with a human on EACH side, driven as the CLI drives it.

    A parks the pairing and a person on A answers it (``_pair``); B's person
    transcribes the code B's own screen shows (``_type_the_code``). Both halves are
    real frames and real files, and the only thing simulated is the two people.
    """
    server_a, server_b, _host, _port = devices
    record, _host, _port = _pair(devices, monkeypatch)

    # A admitted B: the row is written BEFORE the frame that announces it.
    member = record.member(server_b.identity.device_id)
    assert member is not None and member.active
    assert member.added_via == "invite"
    assert set(member.capabilities) == set(types.capabilities_for_role("drive"))
    assert record.invites[0].state == "consumed"

    # B holds the network, the secret and — the gap this slice closed — the
    # INVITER's public key, without which no later handshake could be verified.
    joined = store.load(record.network_id, server_b.root)
    assert joined.epoch == 1
    assert len(joined.active_members()) == 2
    inviter_row = joined.member(server_a.identity.device_id)
    assert inviter_row is not None
    assert inviter_row.public_key == server_a.identity.public_key
    assert store.load_secrets(record.network_id, server_b.root).secret

    # THE QUESTION THAT WAS ASKED names both codes: the inviter's own derivation and
    # the joiner's transcription, which is what makes the inviter's human the
    # comparator rather than a spectator. §5.3: "B transcribes, A compares".
    parked = _pair_answered[record.network_id]
    assert parked["sas"] and len(str(parked["sas"])) == 6
    assert parked["transcribed"] == parked["sas"], "the two sides must derive the same code"
    assert parked["joiner_device_id"] == server_b.identity.device_id
    assert wire.sas_display(str(parked["sas"])) in str(parked["prompt"])
    assert "YOUR screen shows" in str(parked["prompt"])

    events = _events(server_a)
    assert "pairing_awaiting_confirmation" in events
    assert "pairing_confirmed" in events
    assert "member_admitted" in events


def test_a_wrong_transcription_admits_nothing_and_the_same_token_still_works(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The negative half of the joiner's step, and the reason it is no longer terminal.

    The codes disagreed, so the join is refused and no member row appears. What CHANGED
    (cross-host QA, Q-XH-6) is what happens to the token: a mistyped digit used to
    consume it, so the ordinary typo cost a fresh invite minted on the device the
    person is NOT sitting at. It is now forgiven — counted on the invite row, so a
    relay restart cannot reset the budget — and this test drives the whole ceremony
    again with the SAME token to prove the forgiveness is real rather than a comment.
    The burn past the budget is pinned by its own cell below, because that is the half
    that keeps the ~2^20 grind the design refused.

    The inviter's human is never asked on the mistyped attempt — the wrong
    transcription is refused before the pairing is parked, which is the ordering that
    keeps a bad code from becoming a question somebody can answer yes to.
    """
    server_a, server_b, host, port = devices
    record = _init_network(server_a)
    token, envelope = _mint_invite(server_a, record)
    _type_the_code(monkeypatch, code="999999")

    with pytest.raises(types.PairingRefusal) as excinfo:
        _join(server_b, host=host, port=port, token=token, envelope=envelope)

    # THE JOINER'S OWN HALF OF THE REFUSAL. This assertion is the one that was
    # missing: the inviter's invite outcome was checked below and the joiner's
    # path was not, so a call to a helper that lives in a different module
    # (``invite.sas_mismatch_sentence``) reached production and turned every wrong
    # transcription — the normal user path — into an AttributeError instead of a
    # refusal sentence (QA round 1, F-1).
    assert excinfo.value.code == "sas_mismatch"
    assert "digit" in excinfo.value.sentence or "code" in excinfo.value.sentence

    refreshed = store.load(record.network_id, server_a.root)
    assert refreshed.member(server_b.identity.device_id) is None
    # FORGIVEN, NOT BURNED: still ``minted``, one failure counted against it, and its
    # last outcome recorded for the listing that explains what happened.
    assert refreshed.invites[0].state == "minted"
    assert refreshed.invites[0].outcome == "sas_mismatch"
    assert refreshed.invites[0].attempts == 1
    events = _await_event(server_a, "pairing_refused")
    assert "pairing_refused" in events
    assert "pairing_awaiting_confirmation" not in events
    assert not store.record_path(record.network_id, server_b.root).exists()

    # THE SAME TOKEN, A CORRECT CODE, NO NEW INVITE: this is the property the fix is
    # for, and it fails on the tree that consumed the invite above.
    _type_the_code(monkeypatch)
    answered: dict[str, Any] = {}
    failures: list[BaseException] = []

    def _answer_and_record() -> None:
        try:
            row = _answer_confirmation(server_a)
            if row:
                answered.update(row)
        except BaseException as exc:  # noqa: BLE001 — reported below, not swallowed
            failures.append(exc)

    thread = threading.Thread(target=_answer_and_record, daemon=True)
    thread.start()
    try:
        joined = _join(server_b, host=host, port=port, token=token, envelope=envelope)
    finally:
        thread.join(30)
    assert not failures, f"the inviter's human step raised: {failures[0]!r}"
    assert joined is not None, "the same token was refused after one forgiven mistype"

    after = store.load(record.network_id, server_a.root)
    assert after.member(server_b.identity.device_id) is not None
    assert after.invites[0].state == "consumed", "an admitted invite must be spent"
    assert store.load(record.network_id, server_b.root).epoch == 1


def test_a_mistype_burns_the_invite_once_the_forgiving_budget_is_spent(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The half that keeps the grind refused: repeated failures still spend the token.

    ``invite.PAIRING_MAX_FORGIVEN_FAILURES`` is the budget, and it is asserted from the
    constant rather than from a literal here so the test and the rule cannot drift. The
    last failure must leave ``consumed`` behind, and the next attempt with the same
    token must then be refused as already used — otherwise "forgiven" would have become
    "unlimited", which is the ~2^20 online grind ``consume`` exists to prevent.
    """
    server_a, server_b, host, port = devices
    record = _init_network(server_a)
    token, envelope = _mint_invite(server_a, record)
    _type_the_code(monkeypatch, code="999999")

    for attempt in range(invite_mod.PAIRING_MAX_FORGIVEN_FAILURES):
        with pytest.raises(types.PairingRefusal):
            _join(server_b, host=host, port=port, token=token, envelope=envelope)
        row = store.load(record.network_id, server_a.root).invites[0]
        assert row.state == "minted", f"failure {attempt + 1} burned the token early"

    with pytest.raises(types.PairingRefusal):
        _join(server_b, host=host, port=port, token=token, envelope=envelope)
    spent = store.load(record.network_id, server_a.root)
    assert spent.invites[0].state == "consumed"
    assert spent.invites[0].attempts == invite_mod.PAIRING_MAX_FORGIVEN_FAILURES

    # …and the spent token is dead for everyone, including a correct code. It is
    # refused DURING the handshake — before any code is compared — and the listener
    # then closes without explaining itself (an open port that explains is an oracle
    # for token validity), so what arrives is `_join_one`'s own sentence rather than a
    # raised refusal. Asserted as a sentence because that is the honest shape of it.
    _type_the_code(monkeypatch)
    dead = _join(server_b, host=host, port=port, token=token, envelope=envelope)
    assert isinstance(dead, str) and dead, dead
    assert "stopped" in dead, dead
    assert not store.record_path(record.network_id, server_b.root).exists()


def test_a_joiner_slower_than_the_handshake_timeout_still_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q-XH-6's real defect: the LISTENER's wait was a machine's budget, not a human's.

    The frame the inviter waits for carries the code a person read off the other
    device's screen, and the wait was ``min(handshake_timeout_s, ttl)`` — ten seconds
    for a machine's round trip. Both real attempts in the cross-host round took ~15 s
    to read six digits off another screen and lost the pairing AND the invite with it.

    The inviter's budget is SHRUNK here rather than making the test sleep ten seconds:
    the property under test is the RELATION between the two numbers (the human's delay
    outlasting the machine's budget), so the budget is what moves. Pre-fix this times
    out at the budget and no pairing is ever parked; post-fix the human's delay is
    inside the window and the ceremony completes.
    """
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    identity_a = identity.mint(root_a, name="inviter-a")
    identity_b = identity.mint(root_b, name="joiner-b")
    #: The machine's budget, deliberately far below a human's reading time.
    machine_budget_s = 4.0
    server_a = serve_shaped_relay(
        root_a,
        monkeypatch,
        settings=relay.NetworkSettings(
            port=0, listen_address="127.0.0.1", handshake_timeout_s=machine_budget_s
        ),
        identity=identity_a,
        audit=audit_mod.AuditLog(root_a),
    )
    server_b = relay.RelayServer(
        root=root_b,
        settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1"),
        identity=identity_b,
        audit=audit_mod.AuditLog(root_b),
    )
    host, port = server_a.bind()
    server_a.bind_control()
    server_a.start()
    # The joining half resolves the config dir from the AMBIENT environment (that is
    # how a user runs it), so the ambient dir has to be B's.
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root_b))
    human_delay_s = machine_budget_s * 1.5

    def _slow_human(args: Any, derived: str, fingerprint: str) -> str:
        del args, fingerprint
        time.sleep(human_delay_s)
        return derived

    monkeypatch.setattr(net_cli, "_read_code", _slow_human)
    record = _init_network(server_a)
    token, envelope = _mint_invite(server_a, record)
    answered: dict[str, Any] = {}
    failures: list[BaseException] = []

    def _answer_and_record() -> None:
        try:
            row = _answer_confirmation(server_a, timeout=human_delay_s + 20.0)
            if row:
                answered.update(row)
        except BaseException as exc:  # noqa: BLE001 — reported below, not swallowed
            failures.append(exc)

    thread = threading.Thread(target=_answer_and_record, daemon=True)
    thread.start()
    started = time.monotonic()
    try:
        joined = _join(server_b, host=host, port=port, token=token, envelope=envelope)
    finally:
        thread.join(human_delay_s + 25.0)
        server_a.stop()
        server_b.stop()
    elapsed = time.monotonic() - started

    assert not failures, f"the inviter's human step raised: {failures[0]!r}"
    assert elapsed >= human_delay_s, (
        f"the joiner's human was not slow ({elapsed:.2f}s < {human_delay_s:.2f}s), so this "
        "cell did not exercise a delay at all"
    )
    assert joined is not None, (
        f"a {human_delay_s:.1f}s human was refused where the machine's own budget is "
        f"{machine_budget_s:.1f}s: the listener is still timing out on the person"
    )
    after = store.load(record.network_id, server_a.root)
    assert after.member(server_b.identity.device_id) is not None
    assert after.invites[0].state == "consumed"


def test_a_declined_confirmation_admits_nobody_and_burns_the_invite(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The inviter's human says no — the case the inviter's prompt exists for.

    B transcribed correctly, so the joiner's half PASSED; the refusal comes from the
    person on A, which is exactly the interlock: a correct code is not sufficient if
    the other screen did not show the same digits.
    """
    server_a, server_b, _host, _port = devices
    with pytest.raises(Exception):
        _pair(devices, monkeypatch, admit=False)
    record = store.load(_init_network(server_a).network_id, server_a.root)
    assert record.member(server_b.identity.device_id) is None
    events = _events(server_a)
    assert "pairing_awaiting_confirmation" in events
    assert "pairing_refused" in events
    assert "member_admitted" not in events
    assert not store.record_path(record.network_id, server_b.root).exists()


def test_an_unanswered_confirmation_times_out_and_admits_nobody(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nobody answers. The window is the invite's own remaining life, so a short
    invite makes this an ordinary, fast test — and the point is that silence
    refuses: an unanswered question must never admit a device.

    No answering thread is started, which is the whole difference from the tests
    above; the relay is left to its own timeout.
    """
    server_a, server_b, host, port = devices
    record = _init_network(server_a)
    state = store.load_secrets(record.network_id, server_a.root)
    minted = invite_mod.mint(record, state.secret, role="drive", ttl_s=1.0)
    record.invites.append(minted.record)
    store.save(record, server_a.root)
    store.save_invite_token(minted.record.invite_id, minted.token, server_a.root)
    _type_the_code(monkeypatch)
    # Either shape is a refusal and both are honest: the joiner times out reading the
    # answer (``None``), or it reads the abort frame the inviter sends and raises.
    # Which one wins is a race between two timers of the same length, so the test
    # accepts both — and asserts the PROPERTY, that nobody was admitted.
    joined: Any
    try:
        joined = _join(server_b, host=host, port=port, token=minted.token, envelope=minted.envelope)
    except Exception as exc:  # noqa: BLE001 — a refusal, asserted below
        joined = exc

    refreshed = store.load(record.network_id, server_a.root)
    assert (
        refreshed.member(server_b.identity.device_id) is None
    ), "an unanswered confirmation admitted a device"
    assert not isinstance(joined, tuple), f"the join reported success with no answer: {joined!r}"
    # The inviter finishes its own window a moment after the joiner's read gives up
    # (both are the invite's remaining life), and the state change that must land is
    # the inviter's. WHAT THAT STATE IS changed with Q-XH-6: silence is a DELAY, so the
    # invite goes back to ``minted`` with the failure counted rather than being consumed
    # — the person who was too slow retries with the same token instead of asking the
    # other device for a new one. What this cell has always guarded is unchanged and is
    # asserted directly: an unanswered question admits nobody, and the token is not left
    # stuck in ``redeemed`` (the state that is neither usable nor spent).
    deadline = time.time() + 5.0
    while time.time() < deadline:
        refreshed = store.load(record.network_id, server_a.root)
        if refreshed.invites[0].state != "redeemed":
            break
        time.sleep(0.1)
    assert (
        refreshed.invites[0].state == "minted"
    ), f"an abandoned pairing left the token {refreshed.invites[0].state!r}"
    # AND A DELAY SPENDS NO CODE BUDGET (agent review round 1, NIT 2): ``attempts``
    # bounds GUESSES, and nobody guessed — so two slow humans plus one typo can no
    # longer exhaust the three forgiven failures the typo path is meant to have. The
    # outcome field still records what happened, which is what the listing reads.
    assert refreshed.invites[0].attempts == 0, refreshed.invites[0]
    assert refreshed.invites[0].outcome == "timeout", refreshed.invites[0]
    assert "pairing_refused" in _events(server_a)
    # And the parked question was cleaned up rather than left behind holding a code.
    assert store.pending_pairings(server_a.root) == []


def test_a_replayed_invite_is_refused(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server_a, server_b, host, port = devices
    record = _init_network(server_a)
    token, envelope = _mint_invite(server_a, record)
    _type_the_code(monkeypatch)

    def _answer_and_record() -> None:
        _answer_confirmation(server_a)

    thread = threading.Thread(target=_answer_and_record, daemon=True)
    thread.start()
    try:
        assert _join(server_b, host=host, port=port, token=token, envelope=envelope) is not None
    finally:
        thread.join(10)
    # Redeem the SAME token again: the record says consumed, so the listener refuses
    # BEFORE the challenge — silently, so the second attempt does not even learn
    # whether the invite was ever valid. The joiner reports WHAT HAPPENED AT THE
    # HOST rather than a bare failure, which is the best it can do without an
    # oracle: a sentence, not a success (QA round 1, F-4).
    second = _join(server_b, host=host, port=port, token=token, envelope=envelope)
    assert isinstance(second, str), second
    assert not isinstance(second, tuple)
    refreshed = store.load(record.network_id, server_a.root)
    assert len(refreshed.active_members()) == 2
    assert refreshed.invites[0].state == "consumed"
    # THE REFUSAL IS SILENT ON THE WIRE BY DESIGN, AND ITS ROW IS LATE. The listener
    # closes the socket without a frame, which is what the joiner's failure sentence
    # above describes — and the record of that refusal is written on the listener's own
    # handshake thread, so it can still be pending when the caller holds the failure.
    # Measured with a delay injected at ``AuditLog.record``: this cell (and this cell
    # only, of the ones that read ``_events``) reds on the TRAIL READ at 10 s, with the
    # events list ending at ``link_closed``.
    assert net_fixtures.wait_for(
        lambda: "handshake_refused" in _events(server_a)
    ), f"the listener recorded no refusal for the replayed invite: {_events(server_a)}"


def test_an_admission_does_not_revert_what_landed_during_the_human_step(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The listener's admission is a read-modify-write that SPANS the human step.

    THE WIDEST WINDOW IN THE PACKAGE, and the one this cell exists for: the listener
    reads its record, shows a code, waits for a person to compare it, and then writes
    the admission from the copy it read. Everything another writer landed in those
    seconds used to be reverted by that write — and nothing reported it, because the
    file that landed was a well-formed record stamped with a HIGHER sequence than the
    one it clobbered. The writers that land there are the ones whose loss is
    expensive: the heartbeat's endpoint sync, a membership pull, and a peer's epoch
    rotation with the member list that travels with it.

    WHAT THE CELL DOES: the human step is where the concurrent write lands, because
    that IS the window, and the write is the relay's own control op (mint a second
    invite) rather than a store poke — so two real write sites are exercised, not a
    simulation of one. Both must be on disk afterwards, beside the member the pairing
    admitted. Pre-fix the second invite is gone: the admission wrote back the record
    the listener had read before it ever showed the code.
    """
    server_a, server_b, host, port = devices
    record = _init_network(server_a)
    token, envelope = _mint_invite(server_a, record)
    _type_the_code(monkeypatch)
    minted_in_the_window: list[str] = []

    def human_step(self: relay.RelayServer, **kwargs: Any) -> types.PairDecision:
        # WHILE THE HUMANS TALK, another writer lands on the same record.
        minted = self._ctl_invite({"network": record.network_id})  # noqa: SLF001
        minted_in_the_window.append(minted["invite_id"])
        return types.PairDecision(
            invite_id=kwargs["invite_id"],
            decision="admit",
            matched=True,
            reason="",
            answered_by="human",
        )

    monkeypatch.setattr(relay.RelayServer, "_inviter_human_step", human_step)
    joined = _join(server_b, host=host, port=port, token=token, envelope=envelope)
    assert joined is not None and not isinstance(joined, str), joined

    assert minted_in_the_window, "the concurrent writer never ran: no window was exercised"
    after = store.load(record.network_id, server_a.root)
    assert after.invite(minted_in_the_window[0]) is not None, (
        "the admission wrote back the copy it read before the human step, and the "
        "invite another writer minted in that window is gone"
    )
    assert after.member(server_b.identity.device_id) is not None, "the pairing admitted nobody"
    assert len(after.active_members()) == 2
    assert after.invites[0].state == "consumed"


def _drop_after_the_hello(
    server: relay.RelayServer, *, host: str, port: int, envelope: Any
) -> None:
    """A join handshake that sends its hello, takes the challenge, and vanishes.

    It stops at ``read_challenge`` on purpose: that frame is proof the listener got
    PAST ``accept_hello`` and past the invite's own checks, which is the state the
    round-1 defect needed. A helper that returned before the challenge would leave
    this test unable to tell "the invite was not burned" from "the listener never
    looked".
    """
    import socket

    from local_operator.network.identity import mint_instance_id

    sock = socket.create_connection((host, port), timeout=5)
    try:
        handshake = Handshake.new(
            role="dialer",
            identity=server.identity,
            network_id=envelope.network_id,
            epoch=envelope.epoch,
            instance_id=mint_instance_id(),
            session_protocol=net_cli._session_protocol(),  # noqa: SLF001 — the CLI's own value
            mode="join",
            capabilities=list(wire.LINK_CAPABILITIES),
            build={},
        )
        handshake.join_block = {
            "invite_id": envelope.invite_id,
            "joiner_public_key": server.identity.public_key,
            "joiner_name": server.identity.name,
        }
        handshake.send_hello(sock)
        handshake.read_challenge(wire.FrameReader(sock), wire.deadline_in(5.0))
    finally:
        sock.close()


def test_a_hello_that_never_authenticates_does_not_burn_the_invite(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Round-1 MAJOR 3: an unauthenticated hello must not consume a valid invite.

    ``redeemed`` used to be written from the hello alone, so a connection that
    dropped before its auth frame left the invite unusable and the honest device's
    retry was refused ``invite_in_use``. Driven as the wire drives it: a real join
    hello (and a real challenge back, so the listener is past its invite checks),
    then a close, then the SAME token joined properly.
    """
    server_a, server_b, host, port = devices
    record = _init_network(server_a)
    token, envelope = _mint_invite(server_a, record)

    _drop_after_the_hello(server_b, host=host, port=port, envelope=envelope)
    refreshed = store.load(record.network_id, server_a.root)
    assert (
        refreshed.invites[0].state == "minted"
    ), "an unauthenticated hello consumed a valid invite"

    # The honest retry, on the same token, succeeds — which is the whole property.
    _type_the_code(monkeypatch)
    thread = threading.Thread(target=lambda: _answer_confirmation(server_a), daemon=True)
    thread.start()
    try:
        joined = _join(server_b, host=host, port=port, token=token, envelope=envelope)
    finally:
        thread.join(15)
    assert joined is not None and not isinstance(joined, str), joined
    assert store.load(record.network_id, server_a.root).invites[0].state == "consumed"
    assert len(store.load(record.network_id, server_a.root).active_members()) == 2


def test_silent_connections_are_capped_before_authentication(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round-1 MAJOR 1: the pre-auth phase is BOUNDED, so a silent connection
    costs a slot and not an unbounded thread.

    The finding's shape: N connections that send nothing produced N live handshake
    threads and 0 links, because ``max_links`` counts established links and an
    unauthenticated connection has none. Asserted on the wire rather than on a
    thread count: past the cap the socket is closed at the accept, so a read on it
    returns EOF immediately, while a slot-holder's read times out because nothing
    has been written to it and it is still open.
    """
    import socket

    root_a = root / "cap"
    server = serve_shaped_relay(
        root_a,
        monkeypatch,
        settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1", max_handshakes=2),
        identity=identity.mint(root_a, name="cap-device"),
        audit=audit_mod.AuditLog(root_a),
    )
    host, port = server.bind()
    server.bind_control()
    server.start()
    held: list[socket.socket] = []
    try:
        for _ in range(6):
            held.append(socket.create_connection((host, port), timeout=5))
        # Accept order is the connect order on one loopback listener, so the first
        # `max_handshakes` are the ones holding slots.
        for quiet in held[2:]:
            quiet.settimeout(5.0)
            assert quiet.recv(1) == b"", "a connection past the pre-auth cap was held open"
        for busy in held[:2]:
            busy.settimeout(1.0)
            with pytest.raises(TimeoutError):
                busy.recv(1)
    finally:
        for sock in held:
            sock.close()
        server.stop()


def _refusal_was_delivered(sock: Any) -> bool:
    """Has the relay closed this connection? Asked only of a socket ``select`` called readable.

    A refused connection is closed with NO frame — that silence is what keeps the port
    from being a probe oracle — so EOF is the only thing a dropped peer ever observes,
    and it is therefore the event a test can wait on instead of polling the log.
    """
    try:
        return sock.recv(1) == b""
    except OSError:
        return True


def test_a_cap_drop_names_itself_in_the_local_audit(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cap's refusal is SILENT to the peer, so the operator's log is where it has
    to be named.

    A dropped connection gets no reply and no frame — that is what keeps the port
    from being a probe oracle (`_accept_loop`) — so a saturated relay used to leave
    no trace at all on the device that was saturated: the symptom was a peer that
    could not connect while this device reported nothing. ONE record per window, not
    one per connection: the flood below drops four connections and must NOT produce
    four records, because an unauthenticated stranger may not churn the log an
    incident is reconstructed from.
    """
    import select
    import socket

    root_a = root / "cap-audit"
    server = serve_shaped_relay(
        root_a,
        monkeypatch,
        settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1", max_handshakes=2),
        identity=identity.mint(root_a, name="cap-audit-device"),
        audit=audit_mod.AuditLog(root_a),
    )
    host, port = server.bind()
    server.bind_control()
    server.start()
    held: list[socket.socket] = []
    try:
        for _ in range(6):
            held.append(socket.create_connection((host, port), timeout=5))
        # WAIT FOR THE FLOOD TO HAVE HAPPENED, NOT FOR A RECORD TO APPEAR. The old
        # loop polled the audit and broke at the FIRST record it saw, which reads a
        # SNAPSHOT of a flood still in progress: a per-connection implementation
        # passes whenever the read happens to land after one drop, and the assertion
        # could not tell that from a coalescer doing its job (measured — deleting the
        # coalescer left this test GREEN). What the relay actually does to a dropped
        # connection is CLOSE it, silently and with no frame (`_accept_loop`), so EOF
        # on the four sockets that never got a pre-auth slot IS "the drops have been
        # delivered" — and ``_note_handshake_cap`` runs BEFORE that close, so a single
        # read once the four are down sees everything the flood had to say. The two
        # sockets that did get slots stay open: a silent peer holds its slot for
        # ``handshake_timeout_s`` (10 s), well past this wait.
        refused = 0
        deadline = time.time() + 10.0
        while refused < 4 and time.time() < deadline:
            ready, _, _ = select.select(held, [], [], 0.05)
            refused = sum(1 for sock in ready if _refusal_was_delivered(sock))
        # A BACKSTOP, NOT THE ASSERTION: a relay that never delivered the four drops
        # in ten seconds is wedged, and a wedged flood must not be read as a coalescer
        # that behaved.
        assert refused >= 4, f"the server closed only {refused} of the four dropped connections"
        rows: list[dict[str, Any]] = [
            row for row in server.audit.tail(limit=200) if row.get("event") == "handshake_refused"
        ]
        # THE CONTRACT IS A BOUND, NOT A COUNT. The window is
        # ``HANDSHAKE_CAP_NOTICE_S`` wide, so a flood that straddles a boundary is
        # told in two records — but never one per connection, which is the churn this
        # record exists to prevent: FOUR drops, so a count approaching that is that
        # bug. The ``== 1`` this replaces was reading a DIFFERENT defect as a count
        # (below) and so punished a correct coalescer for it.
        assert 1 <= len(rows) < 4, rows
        # AND EVERY ROW MUST BE A DIFFERENT ROW. ``seq`` is stamped once per recorded
        # event, so two rows sharing one is not the coalescer failing at all: it is
        # the WRITER publishing ONE record twice. Measured on this head, that is what
        # the flake was — 3 of 30 isolated runs, both rows identical in ``seq`` and
        # ``ts`` — and it is why the bound above cannot be the only assertion: a
        # duplicated line passes every bound. ``AuditLog.flush`` now takes the
        # payload off the buffer before it opens the file, so a second flusher in
        # that window (the heartbeat, or a reader's ``tail``) finds nothing to
        # re-publish; ``tests/unit/network/test_audit.py`` pins that directly.
        assert len({row["seq"] for row in rows}) == len(rows), rows
        for row in rows:
            assert row["cause"] == "handshake_cap", row
            assert row["outcome"] == "refused", row
            # THE ROW SAYS WHAT IT KNOWS AND WHY IT CANNOT NAME AN ACTOR (Q-R1-4).
            # A cap drop happens before a single frame is read, so there is nothing to
            # attribute — and `their_device` used to carry the socket ADDRESS, which
            # read as a device id to every consumer of this log.
            assert row["detail"]["cause"] == "handshake_cap", row
            assert row["detail"]["mode"] == "unauthenticated", row
            assert row["detail"]["their_addr"].startswith("127.0.0.1:"), row
            # PRESENT AND EMPTY, because nothing is attributable; `unidentified` is what
            # says so. The old row put the socket address here, which every reader of a
            # field named for a device reads as an id.
            assert row["detail"]["their_device"] == "", row
            assert row["detail"]["unidentified"], row
    finally:
        for sock in held:
            sock.close()
        server.stop()


def test_a_member_cannot_send_into_or_close_another_members_stream(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round-1 MINOR 5: the stream-ownership rule holds on send and close, not
    only on push.

    A stream id is unpredictable (``os.urandom``), but unpredictable is not
    unforgeable — a member that learns one from a log, a traceback or a bug must not
    be able to write into, or end, another member's stream. The PUSH path always
    checked ``stream.link is link``; the request path (``net_stream`` with
    ``send``/``close``) did not, so knowing the id was enough.
    """
    root_a = root / "streams"
    server = serve_shaped_relay(
        root_a,
        monkeypatch,
        identity=identity.mint(root_a, name="stream-device"),
        audit=audit_mod.AuditLog(root_a),
    )
    owner = cast(relay.PeerLink, object())
    other = cast(relay.PeerLink, object())
    stream = relay._Stream(  # noqa: SLF001 — the object the ownership rule protects
        stream_id="s_" + "a" * 16,
        session_id="sess-1",
        peer_device_id="d_owner",
        link=owner,
    )
    server._streams[stream.stream_id] = stream  # noqa: SLF001

    assert server._stream_for(owner, stream.stream_id) is stream  # noqa: SLF001
    assert server._stream_for(other, stream.stream_id) is None  # noqa: SLF001
    # The push and closed routes answer "not mine" for a link that does not own it.
    assert server.route_stream_push(other, {"stream": stream.stream_id, "frame": {}}) is False
    assert server.route_stream_closed(other, {"stream": stream.stream_id}) is False
    # A close REQUEST from the wrong link is refused by name rather than obeyed, and
    # with the same refusal an unknown id gets — the answer never confirms whether
    # the id exists on this device.
    with pytest.raises(types.MeshRefusal) as excinfo:
        server._op_stream(  # noqa: SLF001
            other, {"op": "net_stream", "action": "close", "stream": stream.stream_id}
        )
    assert excinfo.value.code == "unknown_stream"
    assert stream.closed is False, "a member closed another member's stream"


# ---------------------------------------------------------------------------
# R1/R4 — a member link, a catalogue, and a refusal
# ---------------------------------------------------------------------------


#: The parked pairing `_pair`'s inviter answered, keyed by network id, so a test can
#: assert on the question that was actually asked and on the codes it carried.
_pair_answered: dict[str, dict[str, Any]] = {}


def _pair(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],
    monkeypatch: pytest.MonkeyPatch,
    *,
    role: str = "drive",
    ttl_s: float = 600.0,
    admit: bool = True,
    settings: relay.NetworkSettings | None = None,
) -> tuple[types.NetworkRecord, str, int]:
    server_a, server_b, host, port = devices
    record = _init_network(server_a)
    state = store.load_secrets(record.network_id, server_a.root)
    minted = invite_mod.mint(record, state.secret, role=role, ttl_s=ttl_s)
    record.invites.append(minted.record)
    store.save(record, server_a.root)
    store.save_invite_token(minted.record.invite_id, minted.token, server_a.root)
    _type_the_code(monkeypatch)
    answered: dict[str, Any] = {}
    failures: list[BaseException] = []

    def _answer_and_record() -> None:
        try:
            row = _answer_confirmation(server_a, admit=admit)
        except BaseException as exc:  # noqa: BLE001 — reported below, not swallowed
            failures.append(exc)
            return
        if row:
            answered.update(row)

    # The two humans run CONCURRENTLY — two people at two keyboards — and the joiner
    # blocks until the inviter's answer reaches it, so answering after the call
    # returns would deadlock on the joiner's own wait.
    thread = threading.Thread(target=_answer_and_record, daemon=True)
    thread.start()
    try:
        joined = _join(
            server_b,
            host=host,
            port=port,
            token=minted.token,
            envelope=minted.envelope,
            settings=settings,
        )
    finally:
        thread.join(10)
    # An exception inside the answering thread would otherwise present as the JOINER
    # timing out minutes later, pointing at the wrong side of the ceremony.
    assert not failures, f"the inviter's human step raised: {failures[0]!r}"
    assert joined is not None
    _pair_answered[record.network_id] = answered
    return store.load(record.network_id, server_a.root), host, port


def test_a_member_dials_and_gets_a_catalogue_reply(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole point of the transport: one relay asks another a question and
    gets an answer, over mutually-authenticated records."""
    server_a, server_b, host, port = devices
    record, _host, _port = _pair(devices, monkeypatch)
    link, reason = server_b.dial(record.network_id, host=f"{host}:{port}", epoch=record.epoch)
    assert link is not None, f"the member handshake failed: {reason}"
    reply = link.request({"op": "net_catalog", "req": 41, "locality": "remote"})
    assert reply is not None, "no reply to net_catalog"
    # A reply is ALWAYS an ack carrying the payload in `detail` (design §10.2) — and
    # that shape is load-bearing: a reply bearing the request's own op name would be
    # read as a new request by the peer, and the two would answer each other forever.
    assert reply["op"] == "ack"
    assert reply["req"] == 41
    detail = reply["detail"]
    assert detail["device"]["device_id"] == server_a.identity.device_id
    assert isinstance(detail["sessions"], list)
    # The catalogue is the READ-THROUGH: it lists session rows, and the relay holds
    # no transcript of its own.
    assert "transcript" not in str(reply)
    assert link.stray_replies == 0, "the link answered something nobody asked"
    link.close("test")


def test_a_read_member_cannot_prompt_through_the_link(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server_a, server_b, host, port = devices
    record, _host, _port = _pair(devices, monkeypatch, role="read")
    link, reason = server_b.dial(record.network_id, host=f"{host}:{port}", epoch=record.epoch)
    assert link is not None, reason
    reply = link.request(
        {
            "op": "net_forward",
            "req": 42,
            "locality": "remote",
            "frame": {"op": "prompt", "command_id": "c1", "text": "do something"},
        }
    )
    assert reply is not None
    assert reply["op"] == "error"
    assert "prompt" in str(reply["message"])
    assert "authorisation_refused" in _events(server_a)
    link.close("test")


def test_a_frame_claiming_local_locality_is_refused_over_the_link(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server_a, server_b, host, port = devices
    record, _host, _port = _pair(devices, monkeypatch)
    link, reason = server_b.dial(record.network_id, host=f"{host}:{port}", epoch=record.epoch)
    assert link is not None, reason
    reply = link.request({"op": "net_catalog", "req": 43, "locality": "local"})
    assert reply is not None and reply["op"] == "error"
    link.close("test")


def test_a_forwarded_session_op_is_refused_with_a_named_sentence(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The session plane belongs to another slice, and this build says so rather than
    silently pretending to have carried the frame."""
    server_a, server_b, host, port = devices
    record, _host, _port = _pair(devices, monkeypatch, role="admin")
    link, reason = server_b.dial(record.network_id, host=f"{host}:{port}", epoch=record.epoch)
    assert link is not None, reason
    reply = link.request(
        {
            "op": "net_forward",
            "req": 44,
            "locality": "remote",
            "frame": {"op": "prompt", "command_id": "c1", "text": "hi", "session_id": "s1"},
        }
    )
    assert reply is not None and reply["op"] == "error"
    assert "session" in str(reply["message"])
    link.close("test")


# ---------------------------------------------------------------------------
# R5 — revocation without visiting the revoked device
# ---------------------------------------------------------------------------


def test_revocation_rotates_the_epoch_and_refuses_the_removed_device_afterwards(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server_a, server_b, host, port = devices
    record, _host, _port = _pair(devices, monkeypatch)
    removed_id = server_b.identity.device_id
    before = store.load_secrets(record.network_id, server_a.root).secret

    result = server_a._ctl_member_rm(  # noqa: SLF001 — the CLI's own control op
        {"network": record.network_id, "device_id": removed_id}
    )
    assert result["epoch"] == 2
    after = store.load_secrets(record.network_id, server_a.root)
    assert after.secret != before
    assert after.previous_epoch == 1

    # CONVERGENCE RULE at the queue: the removed device is offline, so the rotation
    # would have been queued for it — and it is NOT, because it carries the secret.
    assert store.queued_frames(removed_id, server_a.root) == []

    # A FRESH connection from the removed device is refused, which is the whole of
    # R5: no visit, no cooperation, no warning.
    # The refusal is SILENT by design: a closed socket, no error frame, and the
    # reason only in the removing device's own audit record — WHICH IS WHY THE
    # DIALER'S OWN REASON IS NAMED FOR WHAT IT OBSERVED. It used to say
    # ``handshake_failed:ConnectionError``, a transport fact that a removed device's
    # operator could not act on; it now says ``handshake_refused:ConnectionError``,
    # because a peer that accepts the connection and closes it during the handshake
    # is what this protocol's refusal looks like (QA round 3, Q-R3-2). The transport
    # class stays in the suffix: `handshake_refused:TimeoutError` is a peer that never
    # answered, which is a different incident.
    link, reason = server_b.dial(record.network_id, host=f"{host}:{port}", epoch=1)
    assert link is None
    assert reason.startswith("handshake_refused") or reason in ("not_a_member", "epoch_stale")
    assert "member_removed" in _events(server_a)
    assert "handshake_refused" in _events(server_a)


def test_the_removed_device_does_not_learn_the_new_secret(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other half of revocation: a device that cannot authenticate must not be
    handed the key it would need to read new traffic."""
    server_a, server_b, host, port = devices
    record, _host, _port = _pair(devices, monkeypatch)
    old_secret = store.load_secrets(record.network_id, server_b.root).secret
    server_a._ctl_member_rm(  # noqa: SLF001
        {"network": record.network_id, "device_id": server_b.identity.device_id}
    )
    new_secret = store.load_secrets(record.network_id, server_a.root).secret
    assert store.load_secrets(record.network_id, server_b.root).secret == old_secret
    assert new_secret != old_secret


def test_an_untrusted_network_refuses_a_fresh_connection(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A panic is a STATE, not a one-off action: a connection that arrives afterwards
    is refused at handshake step 3."""
    server_a, server_b, host, port = devices
    record, _host, _port = _pair(devices, monkeypatch)
    server_a._ctl_panic({"network": record.network_id})  # noqa: SLF001
    assert store.load(record.network_id, server_a.root).trust == "untrusted"
    link, reason = server_b.dial(record.network_id, host=f"{host}:{port}", epoch=record.epoch)
    assert link is None
    assert "untrusted" not in reason  # never told why: the refusal is a closed socket
    assert "handshake_refused" in _events(server_a)
    # Recovery is explicit and local.
    server_a._ctl_trust({"network": record.network_id, "trust": "active"})  # noqa: SLF001
    assert store.load(record.network_id, server_a.root).trust == "active"


def test_a_second_live_claim_on_one_device_id_evicts_and_audits(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],
) -> None:
    """Copied-key detection: the copy IS the device at the crypto layer, so what the
    design does instead is make it visible — a second live claim outside the restart
    grace window evicts the first, audits it, and flags the member.

    The tracker is driven directly because the wire path would need a second process
    holding a copied key; the FENCE is what is under test.
    """
    server_a = devices[0]
    tracker = server_a.identity_use
    assert tracker.observe("d_copy", instance_id="i_one", link_id="l1", now=1000.0).kind == "new"
    verdict = tracker.observe("d_copy", instance_id="i_two", link_id="l2", now=1060.0)
    assert verdict.kind == "duplicate"
    assert verdict.evicted is not None
    server_a._note_duplicate("d_copy", "i_two")  # noqa: SLF001 — the relay's own audit path
    assert "duplicate_identity" in _events(server_a)


# ---------------------------------------------------------------------------
# F-2 — an endpoint a PEER can dial, without being told the address
# ---------------------------------------------------------------------------


def test_a_paired_device_is_dialable_from_its_record_alone(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each side records an address for the other that `_ensure_link` can dial.

    QA round 1's blocker (F-2), as a property. After a real join each side must
    hold an endpoint for the other that IT never observed, and a relay with nothing
    in memory — the production shape, where the process that pairs is not the
    process that later lists — must form the link by dialling it. Before this, the
    self row was written `endpoints: []` and the only other address ever recorded
    was the observed source address (an ephemeral NAT port), so a paired peer was
    permanently unreachable and every remote-session verb refused.
    """
    server_a, server_b, host, port = devices
    # B IS BOUND AND LISTENING so the reciprocal dial can be proven, not asserted.
    host_b, port_b = server_b.bind()
    server_b.bind_control()
    server_b.start()
    record, _host_a, _port_a = _pair(
        devices,
        monkeypatch,
        settings=relay.NetworkSettings(port=port_b, listen_address="127.0.0.1"),
    )

    # B holds A's endpoint even though B never dialled A: it came from A's welcome.
    joined = store.load(record.network_id, server_b.root)
    inviter_row = joined.member(server_a.identity.device_id)
    assert inviter_row is not None and inviter_row.endpoints
    # F-7: B's durable record advertises B, NOT the address B happened to dial.
    assert joined.listen.get("advertised") == [f"127.0.0.1:{port_b}"]

    # A FRESH RELAY on B's saved root, with nothing in memory, dials A from the row.
    # This is the production path: the relay that pairs is not the relay that lists.
    fresh = serve_shaped_relay(
        server_b.root,
        monkeypatch,
        settings=relay.NetworkSettings(port=port_b, listen_address="127.0.0.1"),
        identity=server_b.identity,
        audit=audit_mod.AuditLog(server_b.root),
    )
    link, reason = fresh._ensure_link_with_reason(  # noqa: SLF001 — the production dial path
        server_a.identity.device_id
    )
    assert link is not None, f"the joiner could not dial the inviter from its record: {reason}"
    link.close("test")

    # ...and the reciprocal: A holds B's DECLARED endpoint, from B's hello.
    a_record = store.load(record.network_id, server_a.root)
    b_row = a_record.member(server_b.identity.device_id)
    assert b_row is not None and b_row.endpoints == [f"{host_b}:{port_b}"], b_row
    back, back_reason = server_a._ensure_link_with_reason(  # noqa: SLF001
        server_b.identity.device_id
    )
    assert back is not None, back_reason
    back.close("test")


# ---------------------------------------------------------------------------
# The relay is not a session owner
# ---------------------------------------------------------------------------


def test_the_relay_never_becomes_a_session_owner() -> None:
    """R1's structural half.

    A relay that owned sessions would allow two writers of one transcript, would
    make `lop network stop` a way to lose work, and would orphan every session it
    owned on a crash. The import graph is where that is enforced, because intention
    is not a barrier.
    """
    forbidden = (
        "session.runtime.serving",
        "session.runtime.process",
        "session.session_factory",
        "local_operator.session.session",
        "session_lease",
    )
    # THE ONE READ THE RELAY MAY MAKE OF THE LEASE, by exact import line.
    # ``session_lease.lease_holder`` reads the claim file WITHOUT acquiring —
    # that non-acquisition is its whole contract (see its docstring) — and the
    # move's source-side retire needs it to refuse moving a session a record-less
    # process is writing. Reading who holds a lease is not holding one. Every
    # OTHER name in that module (``acquire_session_lease``, the reapers) is still
    # forbidden: the carve-out is the literal line, so importing anything beside
    # ``lease_holder`` on it, or importing the module, is an offender again.
    read_only = "from local_operator.session_lease import lease_holder\n"
    offenders: list[str] = []
    for path in Path(__file__).resolve().parents[3].joinpath("local_operator/network").glob("*.py"):
        text = path.read_text(encoding="utf-8").replace(read_only, "")
        for name in forbidden:
            if f"import {name}" in text or f"from local_operator.{name}" in text:
                offenders.append(f"{path.name}: {name}")
    assert offenders == [], f"the relay imports a session owner: {offenders}"


#: The modules the MOVE owns, exempt from the PROXIMITY heuristic below but NOT
#: from the per-call rule (see ``_write_sites_in_text``): moving a session is the one
#: thing the relay does that writes session state, and it does it in staging (outside
#: ``sessions/``) followed by one ``os.replace``, or by recovering a replica into a
#: NEW session directory. ``tests/unit/network/test_mobility.py`` pins that, and
#: ``tests/unit/session/test_no_session_deletion.py`` allow-lists every rename and
#: rmtree in them at the call site.
_MOVE_WRITERS: frozenset[str] = frozenset({"mobility.py", "sync.py"})

#: The ONE module exempt from the bare-name rule below, and the reason: the copy
#: module IS the thing that names a transcript, because naming it is its job. Every
#: other module in the package must not even mention the word.
#:
#: THIS EXEMPTION USED TO COVER ``mobility.py`` TOO, and that is what review round 1
#: found (T3): a ``transcript.jsonl`` truncation added to ``mobility.py`` passed both
#: this guard and ``test_no_session_deletion``, so the two together read as coverage
#: without being it. The scope is now the exact file that needs it.
_CONTENT_NAME_EXEMPT: frozenset[str] = frozenset({"sync.py"})

#: Write shapes that put bytes into a file, keyed to the label the scan reports.
_WRITE_ATTRS: frozenset[str] = frozenset({"write_text", "write_bytes"})


def _open_mode(node: ast.Call) -> str:
    """The mode argument of an ``open`` call, or ``""`` when it is not a literal."""
    mode = ""
    if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
        mode = str(node.args[1].value)
    for keyword in node.keywords:
        if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
            mode = str(keyword.value.value)
    return mode


def _write_shape(node: ast.Call) -> str | None:
    """``"write_text"`` / ``"write_bytes"`` / ``"open(w)"`` / ``"open(a)"``, or None.

    BOTH spellings of ``open``: the builtin (``open(path, "w")``) and the bound
    method (``handle.open``), because a scan that only saw one of them would be the
    same blind spot in a different place.
    """
    func = node.func
    if isinstance(func, ast.Attribute):
        if func.attr in _WRITE_ATTRS:
            return func.attr
        if func.attr == "open":
            mode = _open_mode(node)
            return f"open({mode})" if mode[:1] in ("w", "a", "x") else None
        return None
    if isinstance(func, ast.Name) and func.id in _WRITE_ATTRS:
        return func.id
    if isinstance(func, ast.Name) and func.id == "open":
        mode = _open_mode(node)
        return f"open({mode})" if mode[:1] in ("w", "a", "x") else None
    return None


def _receiver_literals(node: ast.Call) -> set[str]:
    """The string literals in the path a write call targets.

    ``(root / "sessions" / sid / "transcript.jsonl").write_text("")`` yields both
    ``sessions`` and ``transcript.jsonl``; ``dest_path.open("ab")`` yields nothing,
    which is the honest limit of this scan (see the test's docstring).
    """
    target: ast.AST | None = None
    # Narrowed on the attribute itself rather than through a local alias: pyright
    # keeps the narrowing here, and the alias form is the shape that reads as a
    # union member long after the ``isinstance`` that proved otherwise.
    if isinstance(node.func, ast.Attribute):
        target = node.func.value
        if node.func.attr == "open" and node.args:
            target = node.args[0]
    elif isinstance(node.func, ast.Name) and node.func.id == "open" and node.args:
        target = node.args[0]
    if target is None:
        return set()
    return {
        leaf.value
        for leaf in ast.walk(target)
        if isinstance(leaf, ast.Constant) and isinstance(leaf.value, str)
    }


def _receiver_state_name(literals: set[str]) -> str:
    """The session-state name a write receiver's literals name, or ``""``.

    SEGMENTS, not whole strings: the reviewed mutation builds its path out of parts
    (``root / "sessions" / sid / "transcript.jsonl"``) while a shell-shaped one is a
    single literal (``"/tmp/sessions/x/title.json"``), and both have to classify the
    same way. A session-state FILE is preferred over the directory name, because the
    file is the fact a reviewer needs.
    """
    hits: set[str] = set()
    for literal in literals:
        for segment in literal.replace("\\", "/").split("/"):
            if segment in _SESSION_STATE_NAMES or segment == SESSIONS_DIRNAME:
                hits.add(segment)
    named = sorted(name for name in hits if name in _SESSION_STATE_NAMES)
    return named[0] if named else (SESSIONS_DIRNAME if hits else "")


def _write_sites_in_text(text: str, *, module: str) -> list[tuple[str, str]]:
    """``(owner, label)`` for every write whose RECEIVER names session state.

    THE TEETH THIS TEST WAS MISSING. A per-module exception cannot see a new write
    added to an exempt module, which is exactly how a transcript truncation in
    ``mobility.py`` stayed green; this scan is per CALL, so a write that names a
    session directory or a session-state file is reported wherever it appears, in
    whichever function.
    """
    tree = ast.parse(text, filename=module)
    found: list[tuple[str, str]] = []
    stack: list[str] = []

    def visit(node: ast.AST) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            stack.append(node.name)
            for child in ast.iter_child_nodes(node):
                visit(child)
            stack.pop()
            return
        if isinstance(node, ast.Call):
            shape = _write_shape(node)
            if shape is not None:
                offending = _receiver_state_name(_receiver_literals(node))
                if offending:
                    found.append((".".join(stack) or "<module>", f"{shape}:{offending}"))
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return found


def _session_state_names() -> frozenset[str]:
    """Every name the copy set calls session state, plus the directory itself.

    DERIVED, not spelled: a name added to the copy set is a name this scan then
    treats as session state, which is the same spec the copy set itself is
    (§7.2, and see ``test_sync_copy_set.py``).
    """
    from local_operator.network import sync

    return frozenset(set(sync.COPY_SET_NAMES) | set(sync.COPY_SET_TREES) | set(sync.NEVER_COPIED))


#: Computed once at import: the scan is a module-level rule, not a per-test one.
_SESSION_STATE_NAMES: frozenset[str] = _session_state_names()


#: Every write site in ``local_operator/network`` whose receiver names session
#: state, keyed ``module.py::function::label``, with the reason it cannot be a
#: write into a session the relay does not own. EMPTY IS THE STATEMENT: the move's
#: own modules write only into staging, the replica store and files they built
#: themselves, so a row appearing here is a review, not a formality.
_ALLOWED_WRITE_SITES: dict[str, str] = {
    # The move's own boot marker, written into its STAGING directory
    # (``network/staging/<id>/ready.json``, outside ``sessions/``) so a crash before
    # the promote can be settled. It is named in ``sync.EXCLUDED_ENTRIES``, which is
    # why it is in view here at all: the adopt path deletes it on the way in, and
    # recovery removes a stray one left inside a promoted session, so this write can
    # never put it into a session the relay does not own.
    "mobility.py::_destination_move::write_text:ready.json": (
        "writes the move's boot marker into its own staging directory, outside sessions/"
    ),
}


def test_the_relay_writes_no_session_state() -> None:
    """R2's second structural half, stated as a rule a reviewer can check.

    WHAT IT CATCHES: any write call in ``local_operator/network`` whose RECEIVER
    names a session directory or a file the copy set calls session state — a
    truncation of ``transcript.jsonl``, an overwrite of ``title.json``, anything
    added to whichever module. It also keeps the older rule that a module other than
    the copy module must not so much as mention a transcript.

    WHAT IT DOES NOT CATCH, stated rather than implied: a write through an ALIASED
    receiver (``path = root / "sessions" / sid / "transcript.jsonl"`` one function
    away, then ``path.write_text("")``). There is no name in the call to classify, so
    the honest covering guard for that shape is the per-call allow-list in
    ``tests/unit/session/test_no_session_deletion.py`` (whose rows are keyed by
    ``path::function::call``) plus the deletion half of this same rule, and a
    reviewer reading a diff in these two modules should expect to check the writes
    by hand — which is the cost the per-module exemption used to hide.
    """
    offenders: list[str] = []
    for path in sorted(
        Path(__file__).resolve().parents[3].joinpath("local_operator/network").glob("*.py")
    ):
        text = path.read_text(encoding="utf-8")
        if path.name in _MOVE_WRITERS:
            pass
        elif "transcript.jsonl" in text:
            offenders.append(f"{path.name}: names a transcript")
        for owner, label in _write_sites_in_text(text, module=path.name):
            if f"{path.name}::{owner}::{label}" not in _ALLOWED_WRITE_SITES:
                offenders.append(f"{path.name}: {owner} writes session state ({label})")
    assert offenders == [], (
        f"{offenders} — a relay write into session state. Reads of the session plane\n"
        "are fine; a WRITE is the leak this rule exists for. If the write is\n"
        "legitimate, add it to _ALLOWED_WRITE_SITES with the reason it cannot touch a\n"
        "session the relay does not own."
    )


def test_the_write_scan_flags_an_exempt_module_and_a_renamed_function() -> None:
    """The scan's own teeth, on the shape review round 1 used to defeat it.

    PROVE THE TEST CAN STILL FAIL. The mutation that stayed green at
    ``ff04d03f1`` is this line inside ``mobility.py``; it is fed to the classifier
    directly, so this cell fails if the classifier ever stops being per-call (which
    is what a per-module exemption amounts to).
    """
    mutation = (
        "def _lifecycle_on_owner(server, link, frame):\n"
        "    root = server.root\n"
        '    sid = "9f3ac1e0b7d2"\n'
        '    (root / "sessions" / sid / "transcript.jsonl").write_text("")\n'
        "    return {}\n"
    )
    assert _write_sites_in_text(mutation, module="mobility.py") == [
        ("_lifecycle_on_owner", "write_text:transcript.jsonl")
    ]
    # An append through open(), a session directory named literally, and an
    # overwrite of another copy-set name are all the same finding.
    assert _write_sites_in_text(
        'def f():\n    open("/tmp/sessions/x/title.json", "w")\n', module="cli.py"
    ) == [("f", "open(w):title.json")]
    assert (
        _write_sites_in_text('def f():\n    open("/tmp/scratch/x.txt", "w")\n', module="cli.py")
        == []
    )
    assert (
        _write_sites_in_text('def f(sessions):\n    sessions.write_bytes(b"")\n', module="cli.py")
        == []
    )
    # And the shapes this scan is honest about NOT catching: an aliased receiver.
    assert (
        _write_sites_in_text(
            'def f(root, sid):\n    path = root / "sessions" / sid / "transcript.jsonl"\n'
            '    path.write_text("")\n',
            module="mobility.py",
        )
        == []
    )


def test_every_allowed_write_site_has_a_reason_and_a_live_call() -> None:
    """The allow-list cannot rot: each row states why and still matches a call."""
    for key, reason in _ALLOWED_WRITE_SITES.items():
        assert reason.strip(), key
    live: set[str] = set()
    for path in sorted(
        Path(__file__).resolve().parents[3].joinpath("local_operator/network").glob("*.py")
    ):
        for owner, label in _write_sites_in_text(
            path.read_text(encoding="utf-8"), module=path.name
        ):
            live.add(f"{path.name}::{owner}::{label}")
    assert set(_ALLOWED_WRITE_SITES) == live, (
        f"stale rows: {sorted(set(_ALLOWED_WRITE_SITES) - live)}; unlisted live sites: "
        f"{sorted(live - set(_ALLOWED_WRITE_SITES))}"
    )


def _events(server: relay.RelayServer) -> list[str]:
    return [str(record.get("event")) for record in server.audit.tail(limit=500)]


def _await_event(server: relay.RelayServer, event: str, *, timeout: float = 5.0) -> list[str]:
    """The relay's audit events, waiting for ``event`` to land first.

    Several of these cells drive the OTHER device's ceremony through a real socket
    and then assert about the row the RELAY's own thread writes — which it writes
    after it has already answered the joiner, so the cell can observe the refusal
    before the inviter's durable record exists. A relay-backed write is concurrent
    with the test thread by construction; a single read is therefore a read of a
    window the cell does not own.

    Bounded, and it does not weaken the assertion it guards: an event that never
    lands still fails the caller's ``in`` check, five seconds later.

    What each cell's margin actually is, measured on 2026-09-22 while the host ran
    at load ~48 (the probe and its transcripts are not committed; the numbers are
    from `AuditLog.record` and `_events` timestamps):

    * the wrong-transcription refusal (this module's ``pairing_refused``):
      min -0.09 ms over 32 runs, i.e. the row beat the joiner's raise by 90 us at
      its tightest, and the cell reads ~0.2 ms after the raise. That is the one
      that failed in CI, and the reason is here: nothing separates the two sides
      but scheduling.
    * the declined confirmation: -2.5 ms at its worst over 4 runs.
    * the unanswered confirmation: the inviter records 117-329 ms AFTER the joiner
      gives up, and the cell's read landed ~150 ms after the row.

    So only the first one needed the wait when this helper was written, and the two
    slower paths were left alone rather than converted for symmetry.

    THE 2026-09-26 ROUND RE-DERIVED THAT, and the answer held: the decline and the
    timeout are ORDERED, not lucky. Both cells read the row written by the pairing
    DECISION (``relay.py``'s ``pairing_refused``/``pairing_confirmed`` record, written
    before the decision is returned), so the refusal the joiner holds is downstream of
    it; the invite's own ``outcome`` is written later still (``relay.py``),
    after the row. The abort-frame ordering this docstring quotes belongs to the
    ceremony's refusals — a conflict raised while pairing — not to a human's answer,
    and it is why those two paths measure as slow rather than as racy. One cell DID
    fail a re-measurement (the replayed invite, on its TRAIL read, at 10 s of delay
    injected at ``AuditLog.record``) and now calls ``net_fixtures.wait_for`` on the
    assertion's own condition like the cells above rather than reading through here.
    """
    # The poll loop is the package's one wait implementation, so this module does not
    # carry a second copy of one that polls at a different cadence. The default stays
    # 5 s because that is this helper's published contract; the cells that had a real
    # read race call ``net_fixtures.wait_for`` directly, at the package default.
    net_fixtures.wait_for(lambda: event in _events(server), timeout_s=timeout)
    return _events(server)


def test_a_stale_peer_record_is_reaped_and_the_relay_reports_its_links(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],
) -> None:
    server_a = devices[0]
    record = server_a.peer_record()
    assert record.protocol == types.MESH_PROTOCOL_VERSION
    assert record.device_id == server_a.identity.device_id
    assert record.control_port == server_a._control_port  # noqa: SLF001
    assert record.networks == []  # publishing always, even with nothing to publish
    published = store.publish_peer_record(record, server_a.root)
    assert published.exists()
    scanned = store.scan_peer_records(server_a.root)
    assert any(row.pid == record.pid for row, _state in scanned)
    status = server_a.status()
    assert status["device_id"] == server_a.identity.device_id
    assert isinstance(status["links"], list)


# ---------------------------------------------------------------------------
# Definitions over a REAL link (review round 1, BLOCKER 1 and MAJOR 5)
# ---------------------------------------------------------------------------


def _fresh_relay(server: relay.RelayServer) -> relay.RelayServer:
    """A relay on an EXISTING root with nothing in memory — the production shape.

    The process that paired is not the process that later pushes, and a restarted relay
    holds no link at all: exactly the state QA measured after a completed pairing, and
    the reason the old tick (which walked ``links``) never fired. Not started: a
    push-only relay dials out and needs no listener, and a test that starts one more
    listener than it stops is a leak.
    """
    return relay.RelayServer(
        root=server.root,
        settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1"),
        identity=server.identity,
        audit=audit_mod.AuditLog(server.root),
    )


def _edit_fields(**overrides: Any) -> AgentEditFields:
    """``AgentEditFields`` with every field spelled out (pyright requires them all —
    ``Field(None, …)`` is not read as a default), overridden by what a test cares
    about. The same helper, for the same reason, is in ``test_agent_profiles.py``."""
    base: dict[str, Any] = dict(
        name=None,
        description=None,
        tags=None,
        categories=None,
        security_prompt=None,
        hosting=None,
        model=None,
        last_message=None,
        temperature=None,
        top_p=None,
        top_k=None,
        max_tokens=None,
        stop=None,
        frequency_penalty=None,
        presence_penalty=None,
        seed=None,
        current_working_directory=None,
    )
    base.update(overrides)
    return AgentEditFields(**base)


def _dialable_devices(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> types.NetworkRecord:
    """A paired pair whose records let EITHER side dial the other.

    B is bound and started (and advertises that endpoint), so ``_ensure_link`` on A's
    side can reach it from A's own membership row — the shape
    ``test_a_paired_device_is_dialable_from_its_record_alone`` established, reused here
    because every definitions push is a dial from the side that holds the rows.
    """
    server_a, server_b, _host, _port = devices
    _host_b, port_b = server_b.bind()
    server_b.bind_control()
    server_b.start()
    record, _host_a, _port_a = _pair(
        devices,
        monkeypatch,
        settings=relay.NetworkSettings(port=port_b, listen_address="127.0.0.1"),
    )
    return record


def test_a_hostile_row_id_over_a_real_link_installs_nothing_outside_the_root(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BLOCKER 1 through the REAL chokepoint, which is how it was reproduced.

    The pure-function cell in ``test_definitions.py`` plants the same row; this one
    sends it from an admin-capable peer over a real socket, because the reviewer's
    repro was measured that way and the sender's ``origin_id`` is the whole attack.
    Before the guard: a complete agent directory, with a sender-chosen
    ``system_prompt.md``, four levels above ``config/agents/``, while the reply said
    ``installed``.
    """
    from local_operator.network import definitions

    server_a, server_b, _host, _port = devices
    _dialable_devices(devices, monkeypatch)
    dialer = _fresh_relay(server_a)
    link, reason = dialer._ensure_link_with_reason(  # noqa: SLF001 — the production dial path
        server_b.identity.device_id
    )
    assert link is not None, reason
    try:
        reply = link.request(
            {
                "op": "net_definitions",
                "req": 91,
                "locality": "remote",
                "phase": "apply",
                "bundle": {
                    "kind": definitions.BUNDLE_KIND,
                    "version": definitions.BUNDLE_VERSION,
                    "origin_device": server_a.identity.device_id,
                    "agents": [
                        {
                            "kind": "agent",
                            "name": "wire-escape",
                            "origin_id": "../../../../escaped-agent",
                            "created_date": "2026-01-01T00:00:00+00:00",
                            "system_prompt": "WRITER CONTROLLED",
                            "fields": {
                                "name": "wire-escape",
                                "description": "",
                                "tags": [],
                                "categories": [],
                            },
                        }
                    ],
                    "teams": [],
                },
            },
            timeout=30.0,
        )
    finally:
        link.close("test")
    assert reply is not None and reply.get("op") != "error", reply
    detail = reply.get("detail") or {}
    assert detail.get("installed") == [], detail
    assert [row.get("name") for row in detail.get("refused") or []] == ["wire-escape"], detail
    # NOTHING, ANYWHERE the receiving user can write: no escaped directory, no agent
    # row, and no trace of the sender's prompt text.
    assert list(server_b.root.rglob("escaped-agent")) == []
    assert AgentRegistry(server_b.root).list_agents() == []
    assert "WRITER CONTROLLED" not in "".join(
        path.read_text(encoding="utf-8", errors="ignore") for path in server_b.root.rglob("*.md")
    )


def test_the_definitions_cadence_dials_a_member_it_holds_no_link_to(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MAJOR 5: the tick walked the LINK table, so a quiet mesh never synced.

    QA measured the state this cell reproduces: after a completed pairing the inviter
    held ZERO links (a link exists only because an op dialled one), a manual tick
    returned ``[]``, and 150 s later the peer had received nothing — so two of the
    class's three stated benefits were inert unless a create happened to run. It walks
    the member RECORDS instead and lets the push's own dial seam open the link, which
    is what makes "pair a bare node and run workloads" true without a create.
    """
    from local_operator.network import definitions

    server_a, server_b, _host, _port = devices
    _dialable_devices(devices, monkeypatch)
    AgentRegistry(server_a.root).create_agent(
        _edit_fields(name="cadence-agent", description="Reaches the peer with no create.")
    )
    # NOTHING IN MEMORY, which is what a relay that paired and then restarted has.
    fresh_a = _fresh_relay(server_a)
    assert fresh_a.links == {}
    assert AgentRegistry(server_b.root).get_agent_by_name("cadence-agent") is None

    syncer = definitions.DefinitionsSyncer(fresh_a)
    outcomes = syncer.tick()
    assert outcomes, "the tick walked no member at all"
    assert outcomes == [(server_b.identity.device_id, "applied")], outcomes
    assert AgentRegistry(server_b.root).get_agent_by_name("cadence-agent") is not None

    # The interval floor still holds ON THE SAME INSTANCE: a second tick a moment
    # later asks nobody, which is what bounds a mesh of many members.
    assert syncer.tick() == []


# ---------------------------------------------------------------------------
# The incident plane: what the PEERS did, on a real pair
# ---------------------------------------------------------------------------


def _live_link(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[types.NetworkRecord, relay.PeerLink]:
    """A paired pair with B holding a live link to A, and the record."""
    server_a, server_b, host, port = devices
    record = _dialable_devices(devices, monkeypatch)
    link, reason = server_b.dial(record.network_id, host=f"{host}:{port}", epoch=record.epoch)
    assert link is not None, f"the member handshake failed: {reason}"
    return record, link


def test_a_panic_reaches_a_real_peer_and_the_receipt_says_what_it_did(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE BLOCKER, over a real socket (QA round 1 trust & operations, Q-R1-1).

    Three things were wrong at once, and this cell pins all three because each one
    alone produced the same operator-visible lie:

    1. the receiver REFUSED the frame — an admin panic is always exactly one epoch
       ahead of every peer, and the epoch gate had a carve-out for ``net_epoch`` and
       none for ``net_panic``, so no peer ever acted;
    2. the receiver's own ack was written AFTER it closed the link, so even a frame
       it accepted reported nothing back;
    3. the receipt counted links written to, so a broadcast nobody acted on read
       exactly like one that landed.

    What is asserted here is therefore the whole chain: the peer does not refuse it,
    the peer ACTS (its epoch moves to the panicker's and it goes untrusted), and the
    panicking device's receipt carries what the peer reported rather than what it
    sent.
    """
    server_a, server_b, _host, _port = devices
    record, link = _live_link(devices, monkeypatch)
    assert link.network_id == record.network_id
    assert store.load(record.network_id, server_b.root).epoch == 1

    # ``control_dispatch`` answers the control socket's own envelope — the CLI's
    # ``_relay_call`` is what unwraps ``detail`` — so the receipt is read from there.
    receipt = server_a.control_dispatch("net_panic_local", {"network": record.network_id})["detail"]

    # (c) THE RECEIPT DESCRIBES THE PEER. ``broadcast_to`` — a count of writes — is
    # gone, and what replaced it is the peer's own report of what it did.
    assert receipt["sent"] == 1, receipt
    assert receipt["acked"] == 1, receipt
    assert receipt["unacked"] == [] and receipt["refused"] == [], receipt
    assert receipt["ok"] is True, receipt
    row = receipt["peers"][0]
    assert row["device_id"] == server_b.identity.device_id, row
    assert row["outcome"] == "acked", row
    assert row["reported"]["applied"] == "untrusted", row
    assert row["reported"]["epoch_after"] == 2, row
    assert row["reported"]["rotation"] == "untrusted:applied", row

    # (b) THE PEER ACTED: its epoch moved to the panicker's, it holds the new secret,
    # and it refuses peer traffic from now on — the transport's §8.2 duty, which used
    # to be the ONLY half implemented and therefore left the fleet split.
    peer = store.load(record.network_id, server_b.root)
    assert peer.trust == "untrusted", peer.trust
    assert peer.epoch == 2, peer.epoch
    sender = store.load(record.network_id, server_a.root)
    assert sender.trust == "untrusted"
    assert sender.epoch == 2
    assert (
        store.load_secrets(record.network_id, server_b.root).secret
        == store.load_secrets(record.network_id, server_a.root).secret
    ), "the receiver kept the old secret, so recovery would need a reconcile grant"

    # The rows both sides read afterwards.
    received = [
        event for event in server_b.audit.tail(limit=200) if event.get("event") == "panic_received"
    ]
    assert len(received) == 1, received
    assert received[0]["detail"]["epoch_before"] == 1, received
    assert received[0]["detail"]["epoch_after"] == 2, received
    assert received[0]["detail"]["rotation"] == "untrusted:applied", received
    delivered = [
        event
        for event in server_a.audit.tail(limit=200)
        if event.get("event") in ("panic_delivered", "panic_undelivered")
    ]
    assert [event["event"] for event in delivered] == ["panic_delivered"], delivered
    summary = [
        event
        for event in server_a.audit.tail(limit=200)
        if event.get("event") == "panic_broadcast_result"
    ]
    assert summary and summary[-1]["detail"]["acked"] == 1, summary

    # RECOVERY IS ONE LOCAL ACT PER DEVICE, and it needs no reconcile grant: after
    # `trust --active` on both, a fresh handshake comes up AT THE SAME EPOCH. This is
    # the design's §3.1 promise ("the network is coherent again immediately"), and it
    # is what the old behaviour made impossible.
    for server in (server_a, server_b):
        server.control_dispatch(
            "net_trust_local", {"network": record.network_id, "trust": "active"}
        )
    assert store.load(record.network_id, server_a.root).trust == "active"
    assert store.load(record.network_id, server_b.root).trust == "active"
    again, reason = server_b.dial(record.network_id, host=f"{_host}:{_port}", epoch=2)
    assert again is not None, f"the post-panic handshake failed: {reason}"
    assert again.epoch == 2, again.epoch
    again.close("test")


def test_a_panic_receipt_reports_a_member_it_could_not_ask(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A peer this device could not ask is an OUTCOME, not an absence (§2.2/§6.5).

    QA measured the opposite contract: a receipt that read `ok: true` while a peer
    refused the frame. The other half of that lie is silence — a member with no link
    used to contribute NOTHING to the count, so a network of three reported one and
    the third, which is still out there on the old epoch, was invisible.
    """
    server_a, server_b, _host, _port = devices
    record, link = _live_link(devices, monkeypatch)
    link.close("test")
    server_b.stop()

    receipt = server_a.control_dispatch("net_panic_local", {"network": record.network_id})["detail"]

    assert receipt["sent"] == 1, receipt
    assert receipt["acked"] == 0, receipt
    assert receipt["unacked"] == [server_b.identity.device_id], receipt
    assert receipt["ok"] is False, receipt
    assert receipt["peers"][0]["reason"], receipt
    rows = [
        event
        for event in server_a.audit.tail(limit=200)
        if event.get("event") == "panic_undelivered"
    ]
    assert [event["detail"]["outcome"] for event in rows] == ["unacked"], rows


def test_a_refused_leave_is_reported_as_a_refusal_not_as_a_reachable_peer(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE MEASURED RECEIPT (QA round 1 trust & operations, Q-R1-2), reproduced.

    QA drove a `lop network disconnect` that printed ``reachable_peers: 2`` while the
    peer's own log held a ``net_leave`` refused as ``policy`` in the reconcile phase
    — the failure existed, on the peer, and no surface on the acting device said so.
    The refusal is induced here the same way it arose there: this device is one epoch
    behind, so its link authenticates into the reconcile phase, where only
    ``net_reconcile`` and ``ping`` may dispatch.

    Both halves are asserted: the receipt reports the refusal with the peer's OWN
    sentence, and the local act still happens — leaving is never conditional on a
    peer's agreement (design §2.1), it is the REPORT that must not claim otherwise.
    """
    server_a, server_b, host, port = devices
    record = _dialable_devices(devices, monkeypatch)
    state = store.load_secrets(record.network_id, server_a.root)
    # A rotates and tells nobody: B's own record still says epoch 1.
    with store.mutate(record.network_id, server_a.root) as live:
        relay.rotate_epoch(
            live,
            state,
            by=server_a.identity.device_id,
            reason="test",
            root=server_a.root,
        )
    link, reason = server_b.dial(record.network_id, host=f"{host}:{port}", epoch=1)
    assert link is not None, f"the reconcile handshake failed: {reason}"

    receipt = server_b.control_dispatch("net_disconnect", {"network": record.network_id})["detail"]
    assert receipt["sent"] == 1, receipt
    assert receipt["acked"] == 0, receipt
    assert receipt["refused"] == [server_a.identity.device_id], receipt
    assert receipt["ok"] is False, receipt
    assert receipt["reachable_peers"] == 0, receipt
    row = receipt["peers"][0]
    assert row["outcome"] == "refused", row
    assert row["reason"], row
    assert "reconcil" in row["reason"].lower(), row

    # The local half, unconditionally: stopped trusting, secret gone, trail kept.
    assert store.load(record.network_id, server_b.root).trust == "disconnected"
    assert not store.secrets_path(record.network_id, server_b.root).exists()
