"""The eager wire bundle's size policy, as a gate rather than a measurement.

The per-locale bundle exists to stop a client fetching a whole locale it does
not render. That optimisation is only real if one namespace cannot grow back
into the payload: a total budget alone is satisfied by 20 small files and one
80 KiB monster, which is exactly the state this policy was written from
(`wire.settings` was 80,658 B of the 116,849 B English wire set).

Two classes, deliberately explicit:

* every EAGER wire namespace must stay at or below ``EAGER_MAX_BYTES``;
* the deferred set is asserted EXACTLY, so a namespace cannot be moved out of
  the eager ceiling by adding a line to a dict; it takes a reviewed policy
  change, and each deferred namespace has its own ceiling.

A failure here names the two legitimate fixes (split the namespace, or defer it
with a review), because the next person will meet it while adding copy, not
while reading this docstring.
"""

from __future__ import annotations

from local_operator.i18n import catalogues, resolve

#: Per-file ceiling for a namespace that ships in the eager bundle. Chosen
#: well above today's largest eager file (8,873 B in en) because it is a
#: guard against structural growth, not a tight packing target.
EAGER_MAX_BYTES = 32 * 1024

#: The deferred namespace's own ceiling. `wire.settings` is 80,658 B in en and
#: 133,641 B in hi today; the headroom is for ordinary copy edits, and the cap
#: forces the split conversation before the file reaches its next order of
#: magnitude.
DEFERRED_MAX_BYTES = 160 * 1024

#: Total raw bytes of eager wire files per locale. Today: en 36,191 B, fr
#: 7,953 B, hi 20,398 B. The ceiling is per locale, not per bundle, so a new
#: locale cannot arrive with a payload no one measured.
EAGER_TOTAL_MAX_BYTES = 64 * 1024


def _eager_namespaces(locale: str) -> tuple[str, ...]:
    return tuple(
        namespace
        for namespace in catalogues.wire_namespaces(locale)
        if namespace not in catalogues.DEFERRED_NAMESPACES
    )


def test_no_eager_wire_namespace_exceeds_the_per_namespace_ceiling() -> None:
    violations = []
    for locale in resolve.shipped_locales():
        for namespace in _eager_namespaces(locale):
            size = len(catalogues.catalogue_bytes(locale, namespace))
            if size > EAGER_MAX_BYTES:
                violations.append(f"{locale}/{namespace} is {size} B (ceiling {EAGER_MAX_BYTES} B)")
    assert not violations, (
        "A namespace grew past the eager ceiling. Either split it into child "
        "namespaces, or add it to DEFERRED_NAMESPACES in "
        "local_operator/i18n/catalogues.py with a review explaining why no "
        "startup surface needs it: " + "; ".join(violations)
    )


def test_the_deferred_set_is_exactly_the_reviewed_one() -> None:
    assert set(catalogues.DEFERRED_NAMESPACES) == {"wire.settings"}, (
        "The deferred set is a reviewed policy, not a place to park a large "
        "namespace. Adding one requires updating this assertion and the reason "
        "it records."
    )
    for namespace, reason in catalogues.DEFERRED_NAMESPACES.items():
        assert reason.strip(), f"{namespace} needs a recorded reason for deferral"


def test_a_deferred_namespace_stays_within_its_own_ceiling() -> None:
    violations = []
    for locale in resolve.shipped_locales():
        for namespace in catalogues.wire_namespaces(locale):
            if namespace not in catalogues.DEFERRED_NAMESPACES:
                continue
            size = len(catalogues.catalogue_bytes(locale, namespace))
            if size > DEFERRED_MAX_BYTES:
                violations.append(
                    f"{locale}/{namespace} is {size} B (deferred ceiling "
                    f"{DEFERRED_MAX_BYTES} B)"
                )
    assert not violations, (
        "A deferred namespace outgrew its own ceiling; split it before it "
        "becomes an on-demand response no client can afford: " + "; ".join(violations)
    )


def test_the_eager_wire_payload_stays_within_the_total_ceiling() -> None:
    violations = []
    for locale in resolve.shipped_locales():
        total = sum(
            len(catalogues.catalogue_bytes(locale, namespace))
            for namespace in _eager_namespaces(locale)
        )
        if total > EAGER_TOTAL_MAX_BYTES:
            violations.append(f"{locale} is {total} B (ceiling {EAGER_TOTAL_MAX_BYTES} B)")
    assert not violations, (
        "The eager wire payload for a locale grew past its total ceiling; the "
        "per-namespace ceiling is not enough when many small files accumulate: "
        + "; ".join(violations)
    )
