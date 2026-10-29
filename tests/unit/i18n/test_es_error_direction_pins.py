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


def test_a_rejected_key_is_not_a_missing_credential(auth):
    """`key_rejected` is the provider refusing a key that exists."""
    rejected = auth["wire.errors.auth.key_rejected"].lower()
    assert "rechaz" in rejected, (
        f"the provider REJECTED the key; the Spanish must say so: {rejected!r}"
    )
    for absent in ABSENT_WORDS:
        assert absent not in rejected, (
            f"{rejected!r} reads as a MISSING credential ({absent!r}), which sends "
            "the reader to the wrong fix"
        )


def test_a_missing_credential_is_not_a_refusal(auth):
    """`credentials_missing` is the opposite direction: nothing stored at all."""
    missing = auth["wire.errors.auth.credentials_missing"].lower()
    assert "no se encontraron" in missing
    assert "rechaz" not in missing, (
        f"credentials_missing reads as a refusal: {missing!r}"
    )


def test_account_not_found_stays_a_lookup_failure(auth):
    value = auth["wire.errors.auth.account_not_found"].lower()
    assert "no se encontr" in value
    assert "rechaz" not in value, "a missing account is not a rejected account"
    assert "credencial" not in value, "an account lookup failure is not a credential failure"


def test_sign_in_missing_keeps_the_start_again_remedy(auth):
    value = auth["wire.errors.auth.sign_in_missing"].lower()
    assert "ya no está disponible" in value
    assert "empieza de nuevo" in value, (
        "the remedy is 'Start again'; a dead end here leaves the user stuck: "
        f"{value!r}"
    )


def test_store_unreadable_names_why_it_matters(auth):
    value = auth["wire.errors.auth.store_unreadable"].lower()
    assert "no se pudo leer" in value
    assert "censo" in value, (
        "the English says the provider census needs the store; dropping that "
        f"loses why the failure matters: {value!r}"
    )


def test_ttl_range_stays_inclusive_and_keeps_its_placeholder(files):
    value = files["wire.errors.files.sign_ttl"]
    assert "{maximum}" in value
    assert "entre 1 y {maximum}" in value, (
        "the accepted range is inclusive from 1; 'entre' with an explicit upper "
        f"bound is what carries that: {value!r}"
    )
    for exclusive in ("mayor que 1", "más de 1", "menos de {maximum}"):
        assert exclusive not in value.lower()


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

    Feeds the checks above the inversions they exist to reject, so a typo in the
    fence fails here instead of passing forever.
    """
    inverted_rejection = "no se encontraron credenciales para este proveedor"
    assert "rechaz" not in inverted_rejection
    inverted_missing = "el proveedor rechazó las credenciales"
    assert "no se encontraron" not in inverted_missing
    dead_end = "este inicio de sesión ya no está disponible"
    assert "empieza de nuevo" not in dead_end
    exclusive = "ttl_s debe ser mayor que 1 y menos de {maximum}"
    assert "entre 1 y {maximum}" not in exclusive
    wrong_cause = "el archivo es demasiado grande: {mime_type}"
    assert "tipo" not in wrong_cause
    rejected_as_absent = "no se encontró la clave de API de este proveedor"
    assert any(word in rejected_as_absent for word in ABSENT_WORDS)
