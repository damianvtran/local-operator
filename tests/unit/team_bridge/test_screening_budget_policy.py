"""The screening BUDGET POLICY: the shipped ceilings, who may change them, and
what still holds under a bigger one.

What these pin (the 10x raise, 2026-11-05, operator decision on measured demand):

* the shipped set is 400/300/350/200 and the ``Store`` class attributes ARE the
  settings module's constants — one source of truth, no second literal;
* a value that is present but unusable reads its OWN shipped default, and a set
  that contradicts the tier nesting is refused at the WRITE boundary and ignored
  WHOLESALE at the READ boundary: never honoured half, never silently clamped,
  and never an exception out of a receiver;
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


def test_a_contradicted_set_names_every_rule_it_breaks() -> None:
    """The three rules, and the KEY a person has to change, not just a number."""

    assert ScreeningBudget(account=400, sender=900).violations() == (
        ("sender_budget_per_hour", "sender <= account"),
    )
    assert ScreeningBudget(account=400, ambient=351).violations() == (
        ("ambient_budget_per_hour", "ambient <= unaddressed"),
    )
    assert ScreeningBudget(account=400, unaddressed=401).violations() == (
        ("unaddressed_budget_per_hour", "unaddressed <= account"),
    )
    # A set can break more than one rule at once; both are named. ``unaddressed``
    # is set explicitly here so it is not itself a third violation (its default,
    # 350, would be above this 10-unit account).
    both = ScreeningBudget(account=10, ambient=20, unaddressed=10, sender=30)
    assert [leaf for leaf, _ in both.violations()] == [
        "ambient_budget_per_hour",
        "sender_budget_per_hour",
    ]
    # And a valid nesting is silent — including the equality the tiers allow.
    assert ScreeningBudget(account=8, ambient=8, unaddressed=8, sender=8).violations() == ()


def test_a_hand_edited_contradiction_reads_the_whole_shipped_set_and_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """HALF A CONFIG IS NEVER HONOURED, at the READ boundary.

    The writer refuses this combination (below), so the only way it reaches disk
    is a hand edit or an older config. Honouring the three good numbers would put
    the receiver in a state no line of the file describes — the operator's own
    account ceiling silently fenced by a stale sibling — so the whole set falls
    back, once, with the key to change named.
    """

    (tmp_path / "config.yml").write_text(
        json.dumps(
            {
                "values": {
                    "team_bridge": {
                        "screening": {
                            "budget_per_hour": 40,
                            "ambient_budget_per_hour": 30,
                            "unaddressed_budget_per_hour": 35,
                            "sender_budget_per_hour": 900,
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING, logger="local_operator.team_bridge.store"):
        budget = Store(tmp_path).screening_budget
    assert budget == ScreeningBudget()
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "sender_budget_per_hour" in caplog.text
    assert "sender <= account" in caplog.text


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
    assert screening_budget_from_config(str(tmp_path)) == edited


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
