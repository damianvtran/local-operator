"""State pins for the Spanish relay chrome and phone setup (`relay.*`, `mobile.setup`).

Two hazards drove these, both found by the independent W4 audit on this wave:

1. **A fixed adjective met a variable noun.** `usage.used` shipped as
   `{value} {unit} usados`, so `21 solicitudes` rendered `usados` (masculine
   plural against a feminine noun) and `1 crédito` rendered `usados` at one. The
   fix puts the agreement on a `plural` over the COUNT, and the fix itself had a
   second bug: adding `many {… de {unit}}` double-supplied the `de`, because
   `unit_*` already emits `de créditos` at exact multiples of 10⁶ — it rendered
   `1.000.000 de de créditos`. So the pins test the COMPOSED path (unit rendered
   first, as `usage-sheet.tsx` does), not the row string in isolation.

2. **Money and activation states.** A string that makes a user believe they were
   charged, activated, or safely rolled back when the opposite is true is the
   worst thing this surface can say. The pins hold the distinctions the English
   makes: already-active vs activated-by-this-run, unconfirmed vs failed,
   "do not assume rollback", and "setup never purchases credit".

What these pins do NOT cover: idiom, a state promise expressed in a NEW key that
was never pinned, and anything rendered (width/wrapping on a phone). The audit is
the semantic gate; this is the fence.
"""

from __future__ import annotations

import pytest

from local_operator.i18n import catalogues
from local_operator.i18n.runtime import render_message


@pytest.fixture(scope="module")
def shell():
    return catalogues.load_catalogue("es", "relay.shell")


@pytest.fixture(scope="module")
def setup():
    return catalogues.load_catalogue("es", "mobile.setup")


def _compose(shell, n: int, unit_key: str) -> tuple[str, str]:
    """Render a usage row the way the sheet does: unit first, then the row."""
    unit = render_message(shell[unit_key], {"count": n}, "es")
    used = render_message(shell["relay.shell.usage.used"], {"value": n, "unit": unit}, "es")
    left = render_message(shell["relay.shell.usage.remaining"], {"value": n, "unit": unit}, "es")
    return used, left


def test_usage_row_agrees_at_one_and_at_many(shell):
    used_one, left_one = _compose(shell, 1, "relay.shell.usage.unit_credits")
    assert used_one == "Se usó 1 crédito", used_one
    assert left_one == "Queda 1 crédito restante", left_one
    used_many, left_many = _compose(shell, 21, "relay.shell.usage.unit_requests")
    assert used_many == "Se usaron 21 solicitudes", used_many
    assert left_many == "Quedan 21 solicitudes restantes", left_many


def test_exact_million_carries_exactly_one_de(shell):
    """`unit_*` already emits `de créditos` at 10⁶; a second `de` is the defect."""
    used, left = _compose(shell, 1000000, "relay.shell.usage.unit_credits")
    assert used == "Se usaron 1.000.000 de créditos", used
    assert left == "Quedan 1.000.000 de créditos restantes", left
    assert "de de" not in used and "de de" not in left
    # …and a count that is NOT an exact multiple of 10⁶ takes no `de` at all.
    used_15, left_15 = _compose(shell, 1500000, "relay.shell.usage.unit_credits")
    assert used_15 == "Se usaron 1.500.000 créditos", used_15
    assert "de " not in left_15.removeprefix("Quedan "), left_15


def test_unit_less_rows_do_not_dangle_a_de(shell):
    assert render_message(
        shell["relay.shell.usage.remaining_unknown_unit"], {"value": 1}, "es"
    ) == ("Queda 1 restante")
    assert (
        render_message(shell["relay.shell.usage.remaining_unknown_unit"], {"value": 1000000}, "es")
        == "Quedan 1.000.000 restantes"
    )
    assert render_message(shell["relay.shell.usage.used_unknown_unit"], {"value": 1}, "es") == (
        "Se usó 1"
    )
    assert (
        render_message(shell["relay.shell.usage.used_unknown_unit"], {"value": 1000000}, "es")
        == "Se usaron 1.000.000"
    )


def test_already_active_is_not_activated_by_this_run(setup):
    value = setup["mobile.setup.billing_already_active"].lower()
    assert (
        "ya estaba activa" in value
    ), f"the subscription was ALREADY active; this run did not activate it: {value!r}"


def test_unconfirmed_stays_unconfirmed_and_forbids_assuming_rollback(setup):
    value = setup["mobile.setup.billing_unconfirmed"].lower()
    assert "no está confirmado" in value
    assert "no des por hecho" in value, (
        "the English says 'do not assume rollback'; dropping it invites a double "
        f"charge: {value!r}"
    )
    for claim in ("falló", "no se activó", "se revirtió", "se deshizo"):
        assert claim not in value, f"an UNCONFIRMED state claims {claim!r}: {value!r}"


def test_setup_never_purchases_credit(setup):
    value = setup["mobile.setup.balance"].lower()
    assert (
        "nunca compra crédito" in value
    ), f"the consent boundary is that setup never buys credit: {value!r}"


def test_completed_activation_stays_marked_not_undone(setup):
    value = setup["mobile.setup.billing_activated_status"].lower()
    assert (
        "no se ha deshecho" in value
    ), f"the completed activation is explicitly not undone: {value!r}"


def test_the_pins_can_fail(shell):
    """A fence nobody has seen fail is a comment — drive the composed path."""
    unit = render_message(shell["relay.shell.usage.unit_credits"], {"count": 1000000}, "es")
    regression = (
        "{value, plural, one {Se usó # {unit}} "
        "many {Se usaron # de {unit}} other {Se usaron # {unit}}}"
    )
    doubled = render_message(regression, {"value": 1000000, "unit": unit}, "es")
    assert "de de" in doubled, doubled  # the regression the `many` branch caused
    no_agreement = render_message(
        "{value, number} {unit} usados", {"value": 21, "unit": "solicitudes"}, "es"
    )
    assert no_agreement == "21 solicitudes usados", no_agreement
    with pytest.raises(AssertionError):
        assert "Se usó 1 crédito" == no_agreement
