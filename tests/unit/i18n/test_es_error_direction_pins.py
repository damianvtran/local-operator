"""Direction pins for the Spanish `wire.errors.*` surface (es W3).

A failure message has one job: tell the reader what went wrong and what to do
next. The hazard for a translation is not ugliness — it is *direction*. A key
that says the credential was **rejected** must not read as **missing**; a range
must not become exclusive; "start again" must not become a dead end. A reader
who takes the wrong next step at the moment something failed loses more than a
bad sentence costs.

Each pin holds one such direction on the shipped Spanish. They are deliberately
few: the independent audit is the semantic gate, these are the fence around the
classes where a silent reversion would mislead.

What these pins do NOT cover, stated so a green run is not over-read:

* they cannot prove the Spanish is idiomatic, or that the cause is *fully*
  described — only that the specific direction word survived;
* they cannot catch a new failure mode that was never pinned (a new key needs a
  new pin, and noticing that is the audit's job);
* they say nothing about rendered context (width, surrounding prose).
"""

from __future__ import annotations

import pytest

from local_operator.i18n import catalogues

#: Words that present a credential as ABSENT rather than refused.
ABSENT_WORDS = ("no se encontr", "no hay", "falta", "faltan")


@pytest.fixture(scope="module")
def auth():
    return catalogues.load_catalogue("es", "wire.errors.auth")


@pytest.fixture(scope="module")
def files():
    return catalogues.load_catalogue("es", "wire.errors.files")


def _rejection_direction(value: str) -> None:
    """Raise unless a rejected-credential message says REJECTED and not ABSENT."""
    lowered = value.lower()
    assert "rechaz" in lowered, f"the provider REJECTED the key; the Spanish must say so: {value!r}"
    for absent in ABSENT_WORDS:
        assert absent not in lowered, (
            f"{value!r} reads as a MISSING credential ({absent!r}), which sends the "
            "reader to the wrong fix"
        )


def _missing_direction(value: str) -> None:
    """Raise unless a missing-credential message says MISSING and not REFUSED."""
    lowered = value.lower()
    assert "no se encontraron" in lowered, f"nothing is stored; say so: {value!r}"
    assert "rechaz" not in lowered, f"credentials_missing reads as a refusal: {value!r}"


def _sign_in_remedy(value: str) -> None:
    """Raise unless an expired sign-in keeps its start-again instruction."""
    lowered = value.lower()
    assert "ya no está disponible" in lowered, f"the state is 'no longer available': {value!r}"
    assert "empieza de nuevo" in lowered, (
        f"the remedy is 'Start again'; a dead end leaves the user stuck: {value!r}"
    )


def _ttl_range(value: str) -> None:
    """Raise unless the ttl range is inclusive and keeps its placeholder."""
    assert "{maximum}" in value, f"the upper bound is a placeholder: {value!r}"
    assert "entre 1 y {maximum}" in value, (
        "the accepted range is inclusive from 1; 'entre' with an explicit upper "
        f"bound is what carries that: {value!r}"
    )
    for exclusive in ("mayor que 1", "más de 1", "menos de {maximum}"):
        assert exclusive not in value.lower(), f"exclusive wording: {value!r}"


def test_a_rejected_key_is_not_a_missing_credential(auth):
    """`key_rejected` is the provider refusing a key that exists."""
    _rejection_direction(auth["wire.errors.auth.key_rejected"])


def test_a_missing_credential_is_not_a_refusal(auth):
    """`credentials_missing` is the opposite direction: nothing stored at all."""
    _missing_direction(auth["wire.errors.auth.credentials_missing"])



def test_account_not_found_stays_a_lookup_failure(auth):
    value = auth["wire.errors.auth.account_not_found"].lower()
    assert "no se encontr" in value
    assert "rechaz" not in value, "a missing account is not a rejected account"
    assert "credencial" not in value, "an account lookup failure is not a credential failure"


def test_sign_in_missing_keeps_the_start_again_remedy(auth):
    _sign_in_remedy(auth["wire.errors.auth.sign_in_missing"])


def test_store_unreadable_names_why_it_matters(auth):
    value = auth["wire.errors.auth.store_unreadable"].lower()
    assert "no se pudo leer" in value
    assert "censo" in value, (
        "the English says the provider census needs the store; dropping that "
        f"loses why the failure matters: {value!r}"
    )


def test_ttl_range_stays_inclusive_and_keeps_its_placeholder(files):
    _ttl_range(files["wire.errors.files.sign_ttl"])


def test_encoding_failure_is_not_a_conversion_instruction(files):
    value = files["wire.errors.files.html_utf8"].lower()
    assert "utf-8" in value
    assert "no es" in value, f"the text must report invalidity, not a state: {value!r}"
    for instruction in ("conviert", "guarda como", "vuelve a guardar", "cambia la codificación"):
        assert instruction not in value, (
            "the English reports an invalid encoding; it does not instruct a "
            f"conversion: {value!r}"
        )


def test_internal_error_stays_internal(files):
    value = files["wire.errors.files.internal_error"].lower()
    assert "error interno" in value, (
        "an internal server error must not be presented as a user-correctable "
        f"input problem: {value!r}"
    )


def test_file_type_refusals_stay_type_refusals(files):
    for key in (
        "wire.errors.files.audio_type",
        "wire.errors.files.html_type",
        "wire.errors.files.image_type",
        "wire.errors.files.video_type",
    ):
        value = files[key].lower()
        assert "{mime_type}" in value, f"{key} must keep its placeholder"
        assert "tipo" in value, f"{key} must refuse on TYPE, and say so: {files[key]!r}"
        for wrong_cause in ("tamaño", "demasiado grande", "corrupto", "dañado"):
            assert wrong_cause not in value, f"{key} blames the wrong property ({wrong_cause!r})"


def test_the_pins_can_fail():
    """A fence nobody has seen fail is a comment.

    Calls the SAME helpers the pins use, with the exact inversions they exist to
    reject, and requires each to raise. A typo in a helper therefore fails here
    instead of passing forever — which a copy of the assertions would not catch
    (code-review R1-01 on #3157).
    """
    with pytest.raises(AssertionError):
        _rejection_direction("no se encontraron credenciales para este proveedor")
    with pytest.raises(AssertionError):
        _missing_direction("el proveedor rechazó las credenciales")
    with pytest.raises(AssertionError):
        _sign_in_remedy("este inicio de sesión ya no está disponible")
    with pytest.raises(AssertionError):
        _ttl_range("ttl_s debe ser mayor que 1 y menos de {maximum}")
    with pytest.raises(AssertionError):
        _ttl_range("ttl_s debe estar entre 1 y 32768")
    # …and the helpers accept the shipped strings, so the fence is not simply
    # always-raising.
    _rejection_direction("El proveedor rechazó esta clave de API.")
    _missing_direction("No se encontraron credenciales guardadas para este proveedor.")
