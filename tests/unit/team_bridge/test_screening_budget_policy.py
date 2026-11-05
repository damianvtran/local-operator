"""The screening BUDGET POLICY: the shipped ceilings, who may change them, and
what still holds under a bigger one.

What these pin (the 10x raise, 2026-11-05, operator decision on measured demand):

* the shipped set is 400/300/350/200 and the ``Store`` class attributes ARE the
  settings module's constants — one source of truth, no second literal;
* a value that is present but unusable reads its OWN shipped default, and a set
  that contradicts the tier nesting is refused at the WRITE boundary while the
  READ boundary CLAMPS DOWN, never up: a hand edit that lowers the account total
  spends less than the file says at worst, warns once naming each held key, and
  never raises out of a receiver (review round 1, S-2/Q-1 replaced an earlier
  all-four-defaults fallback that turned a tightening edit into a 10x raise);
* the READ path never mutates the operator's config file, however malformed it
  is (review round 1, S-1/Q-2: ``ConfigManager`` renames what it cannot parse);
* resetting ANY one of the four rows restores all four, so the shipped Reset
  button cannot write a set the receiver refuses (review round 1, R-2);
* an edited set takes effect through the real writer and the real receiver, with
  no code change and no restart;
* the tier ordering, the per-sender share and the refill instant are all read
  from the CONFIGURED ceilings, not from the class constants;
* the once-per-episode owner notice and the breaker are untouched by these keys:
  a budget refusal never opens the breaker, and a provider outage opens it at the
  same threshold whatever the ceiling is;
* the shipped ceiling still trips at 400 units, so the default is verified and
  not merely declared.

Tests run at a SMALL explicit ceiling wherever the point is the RULE rather than
the number: 400 reservations say nothing 8 do not, at fifty times the cost. The
one exception is the shipped-ceiling flood below, and it says so.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.team_bridge import settings as tb_settings
from local_operator.team_bridge.settings import ScreeningBudget
from local_operator.team_bridge.store import (
    OwnerReceipt,
    Store,
    StoreRefused,
    screening_budget_from_config,
    state_db_path,
)
from tests.unit.team_bridge.test_screening import (  # noqa: F401 - autouse fixture
    _fixture_models_are_evaluated,
    _owner,
)


def _store_with(config_dir: Path, budget: ScreeningBudget) -> tuple[Store, Any]:
    """An enrolled store at an EXPLICIT ceiling.

    The explicit argument, not a config file, because these tests are about what
    the store does with a given set; :func:`test_an_edited_ceiling_reaches_the_
    receiver_without_a_code_change` is the one that proves a config edit arrives
    there on its own.
    """

    owner = _owner()
    store = Store(config_dir, screening_budget=budget)
    store.commit_enrollment(owner, OwnerReceipt("bridge_consent_v1", "ref-1"))
    return store, owner


def _reserve(store: Store, owner: Any, message_id: str, *, tier: int = 3, sender: str = "") -> str:
    return store.reserve_screening_batch(
        owner,
        [{"message_id": message_id, "sender_account_id": sender}],
        tier=tier,
    )[0]


# ------------------------- (a) the shipped ceilings -------------------------


def test_the_shipped_ceilings_are_the_ten_x_values_at_the_old_relationships() -> None:
    budget = ScreeningBudget()
    assert (budget.account, budget.ambient, budget.unaddressed, budget.sender) == (
        400,
        300,
        350,
        200,
    )
    # The RATIOS moved with the numbers rather than past them: ambient is still
    # three quarters of the account (was 30 of 40), the person reserve is still
    # the last eighth (was 5 of 40) and the per-sender share is still half (was
    # 20 of 40). A raise that broke these would change WHO gets screened first.
    assert budget.ambient * 4 == budget.account * 3
    assert budget.account - budget.unaddressed == budget.account // 8
    assert budget.sender * 2 == budget.account


def test_the_store_attributes_are_the_settings_modules_defaults() -> None:
    """One source of truth: the class carries the SHIPPED defaults, not copies."""

    assert (
        Store.SCREENING_BUDGET_PER_HOUR
        == tb_settings.DEFAULT_TEAM_BRIDGE_SCREENING_BUDGET_PER_HOUR
        == 400
    )
    assert (
        Store.SCREENING_AMBIENT_BUDGET_PER_HOUR
        == tb_settings.DEFAULT_TEAM_BRIDGE_SCREENING_AMBIENT_BUDGET_PER_HOUR
        == 300
    )
    assert (
        Store.SCREENING_UNADDRESSED_BUDGET_PER_HOUR
        == tb_settings.DEFAULT_TEAM_BRIDGE_SCREENING_UNADDRESSED_BUDGET_PER_HOUR
        == 350
    )
    assert (
        Store.SCREENING_SENDER_BUDGET_PER_HOUR
        == tb_settings.DEFAULT_TEAM_BRIDGE_SCREENING_SENDER_BUDGET_PER_HOUR
        == 200
    )
    # ... and a store that has read no config runs at exactly those.
    assert ScreeningBudget().account == Store.SCREENING_BUDGET_PER_HOUR


# --------------------- (b) per-field and cross-field rules ---------------------


@pytest.mark.parametrize("raw", [0, -5, "abc", None, True, False, 2.5, [400], {}])
def test_a_present_but_unusable_field_reads_its_own_default(raw: Any) -> None:
    """Bad input degrades to the shipped ceiling for THAT field; it never raises.

    ``True`` is refused by name because ``bool`` is an ``int`` to Python: it
    would otherwise read as 1, the tightest ceiling there is, from a value
    nobody meant as a number.
    """

    budget = ScreeningBudget.from_values({"budget_per_hour": raw})
    assert budget == ScreeningBudget()


def test_one_unusable_field_does_not_discard_the_others() -> None:
    budget = ScreeningBudget.from_values({"budget_per_hour": 50, "sender_budget_per_hour": "abc"})
    assert (budget.account, budget.sender) == (50, 200)


@pytest.mark.parametrize("values", [None, "off", [], 7])
def test_a_subtree_that_is_not_a_mapping_reads_the_shipped_set(values: Any) -> None:
    assert ScreeningBudget.from_values(values) == ScreeningBudget()


@pytest.mark.parametrize(
    "configured,expected",
    [
        # THE OPERATOR'S OWN TIGHTENING EDIT. Lowering the account total and
        # leaving its slices alone is the single most likely correction a person
        # makes, and it is the case that used to run the SHIPPED 400 an hour
        # behind a settings page still showing 10 (review round 1, S-2 / Q-1).
        ((10, 300, 350, 200), (10, 10, 10, 10)),
        # One tier above the ceiling it is carved out of: held down to its
        # parent, every other number untouched.
        ((400, 500, 350, 200), (400, 350, 350, 200)),
        ((400, 300, 900, 200), (400, 300, 400, 200)),
        ((400, 300, 350, 900), (400, 300, 350, 400)),
        # A VALID hand-edited set is adopted exactly, including the equality the
        # nesting allows and numbers far above the shipped four.
        ((1200, 900, 1000, 600), (1200, 900, 1000, 600)),
    ],
)
def test_a_contradicted_set_is_clamped_down_to_its_parent(
    configured: tuple[int, int, int, int], expected: tuple[int, int, int, int]
) -> None:
    """(account, ambient, unaddressed, sender) in, what the receiver runs out.

    CLAMP DOWN, NEVER UP. A read that spends less than the file states is honest
    about it in one warning; a read that spends MORE is the defect this replaced.
    """

    account, ambient, unaddressed, sender = configured
    budget, held = ScreeningBudget(
        account=account, ambient=ambient, unaddressed=unaddressed, sender=sender
    ).clamped()
    assert (budget.account, budget.ambient, budget.unaddressed, budget.sender) == expected
    # Every difference between the file and the receiver is NAMED, with the value
    # in force — a clamp nobody can see is the same defect in quieter clothing.
    named = {entry.leaf for entry in held}
    assert named == {
        leaf
        for leaf, was, now in (
            ("unaddressed_budget_per_hour", unaddressed, budget.unaddressed),
            ("ambient_budget_per_hour", ambient, budget.ambient),
            ("sender_budget_per_hour", sender, budget.sender),
        )
        if was != now
    }
    for entry in held:
        assert entry.in_force < entry.configured
        assert str(entry.in_force) in entry.describe()


def test_a_valid_nesting_needs_no_correction() -> None:
    """The equality the tiers allow, and the shipped set itself, are untouched."""

    assert ScreeningBudget(account=8, ambient=8, unaddressed=8, sender=8).clamped() == (
        ScreeningBudget(account=8, ambient=8, unaddressed=8, sender=8),
        (),
    )
    assert ScreeningBudget().clamped() == (ScreeningBudget(), ())


def test_an_account_ceiling_above_the_editable_maximum_reads_as_the_maximum() -> None:
    """The one clamp that is not about the nesting: the page's own maximum.

    A number above it cannot be shown or edited back by any surface, so the read
    runs the largest value the product can still lower.
    """

    budget, held = ScreeningBudget(account=10**9).clamped()
    assert budget.account == tb_settings.SCREENING_BUDGET_MAX_READ == 100_000
    assert [entry.leaf for entry in held] == ["budget_per_hour"]


def test_a_hand_edited_tightening_edit_is_honoured_and_warns_once(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A TIGHTENED hand edit runs TIGHT, through the real receiver read.

    The writer refuses a contradicted set (below), but a hand edit can lower the
    account total without touching its slices, and that edit is the one a person
    most plausibly makes. It used to be discarded for the shipped — larger — set,
    so the receiver spent 10x the number the page showed (round 1, S-2/Q-2). Now
    the slices come down with it and one warning names each held key and the
    value in force.
    """

    (tmp_path / "config.yml").write_text(
        json.dumps(
            {
                "values": {
                    "team_bridge": {
                        "screening": {
                            "budget_per_hour": 10,
                            "ambient_budget_per_hour": 300,
                            "unaddressed_budget_per_hour": 350,
                            "sender_budget_per_hour": 200,
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING, logger="local_operator.team_bridge.store"):
        budget = Store(tmp_path).screening_budget
    assert (budget.account, budget.ambient, budget.unaddressed, budget.sender) == (
        10,
        10,
        10,
        10,
    )
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1, caplog.text
    for leaf in (
        "unaddressed_budget_per_hour",
        "ambient_budget_per_hour",
        "sender_budget_per_hour",
    ):
        assert leaf in caplog.text
    assert "10 in force" in caplog.text
    assert "in force: 10/10/10/10" in caplog.text
    # ONE line per distinct correction: a second construction of the same file,
    # and a Store built from the same directory by another caller, stay silent.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="local_operator.team_bridge.store"):
        assert screening_budget_from_config(tmp_path) == budget
        assert Store(tmp_path).screening_budget == budget
    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


def test_a_config_read_never_moves_a_malformed_config_aside(tmp_path: Path) -> None:
    """THE READ PATH DOES NOT MUTATE THE OPERATOR'S FILE (round 1, S-1 / Q-2).

    ``ConfigManager``'s constructor renames a config it cannot parse to
    ``config.yml.bad.<stamp>`` and prints to stderr. Right at startup; wrong from
    a receiver poll that can catch an editor's non-atomic save mid-write, where it
    would move the live config — every setting with it — aside from background
    work. ``config.read_config_values`` is the reader that does not, so this test
    pins four things: the shipped ceilings, the file still there byte-identical,
    no ``.bad`` sibling, and that a VALID file is still read (the safe reader is
    not a silent no-op).
    """

    path = tmp_path / "config.yml"
    torn = "values: [unclosed\n"
    path.write_text(torn, encoding="utf-8")
    assert Store(tmp_path).screening_budget == ScreeningBudget()
    assert path.exists() and path.read_text(encoding="utf-8") == torn
    assert not list(tmp_path.glob("config.yml.bad*"))

    # A document that parses but is not a mapping takes the same path.
    path.write_text("- just\n- a list\n", encoding="utf-8")
    assert Store(tmp_path).screening_budget == ScreeningBudget()
    assert path.exists() and not list(tmp_path.glob("config.yml.bad*"))

    # The control: the safe reader still ANSWERS a readable file.
    path.write_text(
        json.dumps(
            {
                "values": {
                    "team_bridge": {
                        "screening": {
                            "budget_per_hour": 12,
                            "ambient_budget_per_hour": 9,
                            "unaddressed_budget_per_hour": 10,
                            "sender_budget_per_hour": 6,
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    assert Store(tmp_path).screening_budget == ScreeningBudget(
        account=12, ambient=9, unaddressed=10, sender=6
    )


@pytest.mark.parametrize(
    "reset_key",
    [
        "team_bridge.screening.budget_per_hour",
        "team_bridge.screening.unaddressed_budget_per_hour",
        "team_bridge.screening.ambient_budget_per_hour",
        "team_bridge.screening.sender_budget_per_hour",
    ],
)
def test_resetting_any_screening_row_restores_the_whole_set(tmp_path: Path, reset_key: str) -> None:
    """Reset ANY of the four and ALL FOUR return to the shipped defaults.

    ``reset_setting`` never runs ``validate`` (there is nothing to validate when
    a value is deleted), so resetting one ceiling to its default used to leave a
    contradicted set behind: the page showed 900/800 while the receiver ran the
    shipped four (round 1, R-2). The four are one nested set, so the reset is one
    action too.
    """

    from local_operator import settings_io
    from local_operator.config import ConfigManager

    manager = ConfigManager(tmp_path)
    rows = {setting.key: setting for setting in settings_io.SETTINGS}
    # A VALID set far above the defaults, so each of the four has something to
    # reset and the assertion cannot pass on "the key was already absent".
    # Outer-first, because each write is judged against the set stored so far.
    for key, value in (
        ("team_bridge.screening.budget_per_hour", 1000),
        ("team_bridge.screening.unaddressed_budget_per_hour", 900),
        ("team_bridge.screening.ambient_budget_per_hour", 800),
        ("team_bridge.screening.sender_budget_per_hour", 200),
    ):
        settings_io.write_setting(manager, rows[key], value)
    assert Store(tmp_path).screening_budget == ScreeningBudget(
        account=1000, ambient=800, unaddressed=900, sender=200
    )

    settings_io.reset_setting(manager, rows[reset_key])

    # The whole subtree is gone, not just the row that was reset...
    stored = ConfigManager(tmp_path).get_nested_value(("team_bridge", "screening"), {})
    assert stored == {}, stored
    # ... so the receiver and the page agree on the shipped set again.
    assert Store(tmp_path).screening_budget == ScreeningBudget()


def test_a_hand_edited_valid_set_is_adopted_wholesale(tmp_path: Path) -> None:
    """The other side of the boundary: a valid hand edit IS the live ceiling."""

    (tmp_path / "config.yml").write_text(
        json.dumps(
            {
                "values": {
                    "team_bridge": {
                        "screening": {
                            "budget_per_hour": 12,
                            "ambient_budget_per_hour": 9,
                            "unaddressed_budget_per_hour": 10,
                            "sender_budget_per_hour": 6,
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    edited = ScreeningBudget(account=12, ambient=9, unaddressed=10, sender=6)
    assert Store(tmp_path).screening_budget == edited
    # A STRING config dir reads the same set: ``Store`` is typed ``Path`` but the
    # reader must not answer the shipped ceilings because a caller passed str —
    # that used to land in its never-raises guard and look like "no config".
    assert screening_budget_from_config(str(tmp_path)) == edited  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "key,value,fragment",
    [
        ("team_bridge.screening.sender_budget_per_hour", 401, "per-sender ceiling"),
        ("team_bridge.screening.ambient_budget_per_hour", 351, "agent chatter ceiling"),
        ("team_bridge.screening.unaddressed_budget_per_hour", 401, "unaddressed ceiling"),
    ],
)
def test_the_writer_refuses_a_contradicted_set_and_names_the_rule(
    tmp_path: Path, key: str, value: int, fragment: str
) -> None:
    """WRITE boundary: refuse, do not clamp.

    A silent clamp would store a number the page then shows forever while the
    runtime uses another one, so the write fails with the rule it would break
    while the user is still looking at the command that caused it.
    """

    from local_operator import settings_io
    from local_operator.config import ConfigManager

    manager = ConfigManager(tmp_path)
    rows = {setting.key: setting for setting in settings_io.SETTINGS}
    with pytest.raises(ValueError) as refused:
        settings_io.write_setting(manager, rows[key], value)
    message = str(refused.value)
    assert fragment in message
    assert str(value) in message


def test_the_writer_still_accepts_a_valid_set(tmp_path: Path) -> None:
    from local_operator import settings_io
    from local_operator.config import ConfigManager

    manager = ConfigManager(tmp_path)
    rows = {setting.key: setting for setting in settings_io.SETTINGS}
    settings_io.write_setting(manager, rows["team_bridge.screening.sender_budget_per_hour"], 400)
    settings_io.write_setting(manager, rows["team_bridge.screening.budget_per_hour"], 400)


# ------------- (c) an edited ceiling reaches the receiver, unchanged -------------


def test_an_edited_ceiling_reaches_the_receiver_without_a_code_change(tmp_path: Path) -> None:
    """The proof that "configurable" means what it says.

    Written through the REAL settings writer (the path ``/settings``, the desktop
    app and ``lop config edit`` all funnel through), then read by a ``Store``
    built the way production builds one — from the config dir alone.
    """

    from local_operator import settings_io
    from local_operator.config import ConfigManager

    manager = ConfigManager(tmp_path)
    rows = {setting.key: setting for setting in settings_io.SETTINGS}
    # One valid step at a time (each write is judged against the stored set):
    # chatter down, sender share down, unaddressed down, account last.
    for key, value in (
        ("team_bridge.screening.ambient_budget_per_hour", 9),
        ("team_bridge.screening.sender_budget_per_hour", 6),
        ("team_bridge.screening.unaddressed_budget_per_hour", 10),
        ("team_bridge.screening.budget_per_hour", 12),
    ):
        settings_io.write_setting(manager, rows[key], value)

    edited = ScreeningBudget(account=12, ambient=9, unaddressed=10, sender=6)
    assert Store(tmp_path).screening_budget == edited

    # And it is ENFORCED, not just read: twelve person-addressed units fit and
    # the thirteenth is refused. Under the shipped 400 it would not refuse until
    # 401, so this cannot pass on a constant.
    store, owner = _store_with(tmp_path, Store(tmp_path).screening_budget)
    for i in range(edited.account):
        _reserve(store, owner, f"bmsg_{i}")
    with pytest.raises(StoreRefused) as refused:
        _reserve(store, owner, "bmsg_over")
    assert refused.value.code == "budget_exhausted"


# ------------------- (d) units, tier ordering, per-sender share -------------------


def test_ambient_chatter_stops_at_its_share_of_the_configured_ceiling(tmp_path: Path) -> None:
    budget = ScreeningBudget(account=8, ambient=6, unaddressed=7, sender=4)
    store, owner = _store_with(tmp_path, budget)
    for i in range(budget.ambient):
        _reserve(store, owner, f"bmsg_amb{i}", tier=0)
    with pytest.raises(StoreRefused) as refused:
        _reserve(store, owner, "bmsg_amb_over", tier=0)
    assert refused.value.code == "ambient_exhausted"
    # The units the chatter could not touch are still there for a person.
    for i in range(budget.account - budget.ambient):
        _reserve(store, owner, f"bmsg_dm{i}")
    with pytest.raises(StoreRefused) as full:
        _reserve(store, owner, "bmsg_dm_over")
    assert full.value.code == "budget_exhausted"


def test_tiers_zero_to_two_stop_at_the_unaddressed_ceiling(tmp_path: Path) -> None:
    budget = ScreeningBudget(account=8, ambient=6, unaddressed=7, sender=4)
    store, owner = _store_with(tmp_path, budget)
    for i in range(budget.ambient):
        _reserve(store, owner, f"bmsg_a{i}", tier=0)
    # ONE more unaddressed unit: an agent's mention or a person's room line.
    _reserve(store, owner, "bmsg_room", tier=1)
    with pytest.raises(StoreRefused) as refused:
        _reserve(store, owner, "bmsg_room_over", tier=1)
    assert refused.value.code == "unaddressed_exhausted"
    # An AGENT addressing the agent is tier 2: the same fence, not a way past it.
    with pytest.raises(StoreRefused) as agent_refused:
        _reserve(store, owner, "bmsg_agent", tier=2)
    assert agent_refused.value.code == "unaddressed_exhausted"


def test_only_a_person_addressing_the_agent_may_use_the_last_unit(tmp_path: Path) -> None:
    budget = ScreeningBudget(account=8, ambient=6, unaddressed=7, sender=4)
    store, owner = _store_with(tmp_path, budget)
    for i in range(budget.unaddressed):
        _reserve(store, owner, f"bmsg_fill{i}", tier=1)
    # The eighth unit exists and is reserved: reachable only at tier 3.
    _reserve(store, owner, "bmsg_dm")
    with pytest.raises(StoreRefused) as full:
        _reserve(store, owner, "bmsg_dm_2")
    assert full.value.code == "budget_exhausted"


def test_one_sender_cannot_spend_more_than_its_share_while_others_still_can(
    tmp_path: Path,
) -> None:
    budget = ScreeningBudget(account=8, ambient=6, unaddressed=7, sender=4)
    store, owner = _store_with(tmp_path, budget)
    for i in range(budget.sender):
        _reserve(store, owner, f"bmsg_loud{i}", sender="acc_loud")
    with pytest.raises(StoreRefused) as refused:
        _reserve(store, owner, "bmsg_loud_over", sender="acc_loud")
    assert refused.value.code == "sender_exhausted"
    # Other senders, and a call of unknown provenance, still reserve.
    _reserve(store, owner, "bmsg_quiet", sender="acc_quiet")
    _reserve(store, owner, "bmsg_unknown")


def test_a_re_delivered_message_stops_at_its_senders_ceiling_with_no_owner_notice(
    tmp_path: Path,
) -> None:
    """ONE sender re-delivering ONE message: the SENDER share stops it first.

    The repeat-screening finding in the round-1 review (a real message was
    screened 155 times over 54 hours) makes this the shape that actually spends
    an hour: not many messages, but one message arriving again and again from the
    same sender. Two properties have to hold, so this pins both.

    1. The PER-SENDER ceiling refuses it — 200 of the shipped 400 — and not the
       account's, so one looping correspondent can never eat the hour that other
       people, and the owner's own messages, need.
    2. NO OWNER NOTIFICATION follows. Sender exhaustion is that sender's problem
       by design: the account still has half its hour, nothing is wrong that the
       owner could act on, and a notice per looping correspondent would train
       them to ignore the one that matters. The notify path belongs to the
       ACCOUNT ceiling (`claim_budget_notification` is the gate every caller
       raises it through, so a ``None`` here IS the assertion that nothing is
       raised).

    The shipped set is deliberate rather than a small explicit one: the numbers
    are the assertion — the refusal lands at exactly half the account ceiling.
    """

    store, owner = _store_with(tmp_path, ScreeningBudget())
    for _ in range(Store.SCREENING_SENDER_BUDGET_PER_HOUR):
        _reserve(store, owner, "bmsg_one_message", sender="acc_loud")

    with pytest.raises(StoreRefused) as refused:
        _reserve(store, owner, "bmsg_one_message", sender="acc_loud")
    assert refused.value.code == "sender_exhausted"

    # The account is HALF spent, so the account ceiling is not what refused it:
    # a different sender, and an unidentified call, both still reserve.
    assert Store.SCREENING_SENDER_BUDGET_PER_HOUR * 2 == Store.SCREENING_BUDGET_PER_HOUR
    _reserve(store, owner, "bmsg_one_message", sender="acc_quiet")
    _reserve(store, owner, "bmsg_one_message")

    # Nothing to tell the owner, and no episode opened for them to be told about.
    assert store.claim_budget_notification(owner) is None
    assert store.screening_breaker_state(owner) is None


# ----------------------------- (e) refill timing -----------------------------


def test_the_refill_instant_follows_the_configured_ceiling(tmp_path: Path) -> None:
    """The retry a hold waits for is the OLDEST unit + one hour, at any ceiling.

    Discriminating by construction: the oldest unit is a tier-1 call while the
    two ambient units are newer. A tier-0 caller waits for the ambient pool only
    while the ACCOUNT window still has room; here the window is full (three
    units, ceiling three, CONFIGURED), so the answer must be the oldest unit of
    all. Reading the class constant (400) instead would leave the window "not
    full", point at the ambient pair and retry an hour early.
    """

    store, owner = _store_with(
        tmp_path, ScreeningBudget(account=3, ambient=2, unaddressed=3, sender=3)
    )
    _reserve(store, owner, "bmsg_oldest", tier=1)
    _reserve(store, owner, "bmsg_amb_1", tier=0)
    _reserve(store, owner, "bmsg_amb_2", tier=0)
    now = time.time()
    conn = sqlite3.connect(state_db_path(tmp_path))
    try:
        aged = (("bmsg_oldest", 300.0), ("bmsg_amb_1", 200.0), ("bmsg_amb_2", 100.0))
        for message_id, age in aged:
            conn.execute(
                "UPDATE screening_attempts SET at = ? WHERE message_id = ?",
                (now - age, message_id),
            )
        conn.commit()
    finally:
        conn.close()
    refill = store.screening_budget_refill_at(owner, tier=0)
    assert refill is not None
    assert refill == pytest.approx(now - 300.0 + 3600.0, abs=2.0)


# ------------------- (f) one owner notice per exhaustion episode -------------------


def test_the_owner_is_notified_once_per_episode_at_any_ceiling(tmp_path: Path) -> None:
    budget = ScreeningBudget(account=4, ambient=3, unaddressed=3, sender=2)
    store, owner = _store_with(tmp_path, budget)
    for i in range(budget.account):
        _reserve(store, owner, f"bmsg_{i}")
    with pytest.raises(StoreRefused) as refused:
        _reserve(store, owner, "bmsg_over")
    assert refused.value.code == "budget_exhausted"
    first = store.claim_budget_notification(owner)
    assert first is not None
    assert store.claim_budget_notification(owner) is None
    # A hold keeps the episode open: a second refusal is still the same episode.
    store.persist_hold(owner, "bmsg_hold", "budget_exhausted", retry_at=0.0)
    assert store.close_budget_episode_if_drained(owner) is False
    with pytest.raises(StoreRefused):
        _reserve(store, owner, "bmsg_again")
    assert store.claim_budget_notification(owner) is None
    # Drained: the episode ends, and the next exhaustion is a NEW one, so the
    # next notice is unread.
    store.delete_hold(owner, "bmsg_hold")
    assert store.close_budget_episode_if_drained(owner) is True
    time.sleep(0.01)
    with pytest.raises(StoreRefused):
        _reserve(store, owner, "bmsg_new_episode")
    second = store.claim_budget_notification(owner)
    assert second is not None and second > first


# ------------------------ (g) the breaker is not the budget ------------------------


def test_a_budget_refusal_never_opens_the_breaker(tmp_path: Path) -> None:
    store, owner = _store_with(
        tmp_path, ScreeningBudget(account=2, ambient=2, unaddressed=2, sender=1)
    )
    _reserve(store, owner, "bmsg_1")
    _reserve(store, owner, "bmsg_2")
    with pytest.raises(StoreRefused):
        _reserve(store, owner, "bmsg_3")
    assert store.screening_breaker_state(owner) is None


def test_a_provider_outage_opens_the_breaker_at_the_same_threshold_at_any_ceiling(
    tmp_path: Path,
) -> None:
    """The breaker's threshold is its own: it reads no budget value.

    Same count of provider failures opens it at the smallest ceiling we can
    configure as at the shipped one — so a 10x budget did not move the thing
    that stops a BROKEN route, and the two bounds fail independently.
    """

    owner = _owner()
    for name, budget in (
        ("small", ScreeningBudget(account=2, ambient=1, unaddressed=1, sender=1)),
        ("shipped", ScreeningBudget()),
    ):
        directory = tmp_path / name
        directory.mkdir()
        store = Store(directory, screening_budget=budget)
        store.commit_enrollment(owner, OwnerReceipt("bridge_consent_v1", f"ref-{name}"))
        retry_at = 0.0
        for _ in range(Store.BREAKER_THRESHOLD):
            retry_at = store.record_screening_failure(owner)
        assert retry_at > time.time(), name
        state = store.screening_breaker_state(owner)
        assert state is not None and int(state["failures"]) == Store.BREAKER_THRESHOLD, name


# ------------------------- (h) the shipped ceiling itself -------------------------


def test_the_shipped_ceiling_trips_at_400_units(tmp_path: Path) -> None:
    """The DEFAULT ceiling, verified rather than declared.

    The only test here that spends the shipped number of rows (400 reservations,
    ~1 s): every other test proves a RULE, which a small ceiling proves just as
    well — but a default nobody trips is a default nobody has checked.
    """

    store, owner = _store_with(tmp_path, ScreeningBudget())
    for i in range(Store.SCREENING_BUDGET_PER_HOUR):
        _reserve(store, owner, f"bmsg_{i}")
    with pytest.raises(StoreRefused) as refused:
        _reserve(store, owner, "bmsg_over")
    assert refused.value.code == "budget_exhausted"
