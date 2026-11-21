"""The eager wire bundle's size policy, as a gate rather than a measurement.

The per-locale bundle exists to stop a client fetching a whole locale it does
not render. That optimisation is only real if one namespace cannot grow back
into the payload: a total budget alone is satisfied by 20 small files and one
80 KiB monster, which is exactly the state this policy was written from
(`wire.settings` was 80,658 B of the 116,849 B English wire set).

Three classes, deliberately explicit:

* every EAGER wire namespace must stay at or below ``EAGER_MAX_BYTES``;
* the deferred set is asserted EXACTLY, so a namespace cannot be moved out of
  the eager ceiling by adding a line to a dict; it must carry a reviewed reason
  (and matching entry in ``catalogues.DEFERRED_NAMESPACES``) and it must still
  fit its own ceiling;
* a locale's EAGER total must stay at or below ``EAGER_TOTAL_MAX_BYTES``.

The totals are RAW UTF-8 catalogue bytes, not transfer bytes: the route serves
them with no content encoding today (``content-encoding=None``; no
``GZipMiddleware``), so this is a structural guard on what the bundle carries,
not a compression budget. Compressed, the eager bodies land somewhere between
a fifth and a third of their raw size depending on the script, and hi — the
largest raw payload — compresses SMALLER than en, which is why a compression
ceiling cannot express this policy. No exact transfer figure is asserted here:
it would pin a serializer and a gzip level this test does not control.

A failure here names the two legitimate fixes (split the namespace, or defer it
with a review), because the next person will meet it while adding copy, not
while reading this docstring.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator.i18n import catalogues, resolve

#: Per-file ceiling for a namespace that ships in the eager bundle. Chosen
#: well above today's largest eager wire file (9,534 B, `wire.errors.desktop`
#: in en) because it is a guard against structural growth, not a tight packing
#: target.
EAGER_MAX_BYTES = 32 * 1024

#: Every deferred namespace's own ceiling. `wire.settings` is the largest at
#: 82,959 B in en and 138,430 B in hi today; the headroom is for ordinary copy
#: edits, and the cap forces the split conversation before the file reaches its
#: next order of magnitude.
DEFERRED_MAX_BYTES = 160 * 1024

#: Total raw bytes of eager wire files per locale. The ceiling is per locale, not
#: per bundle, so a new locale cannot arrive with a payload no one measured.
#:
#: Measured basis (2026-11-15): merged main c7f96b4ec has 26 eager namespaces
#: in each of 8 shipped locales; hi is largest at 67,674 raw UTF-8 bytes.
#: The 72 KiB ceiling leaves 6,054 B for ordinary copy edits while bringing the
#: split-or-defer conversation back before structural growth accumulates.
#: It stays a growth guard, not a packing target or a compressed-byte budget.
#:
#: Raised 72 -> 76 KiB (77,824 B) when wire.voice was translated: wire.voice
#: (37 keys, ~7 KB in hi) is eager and cannot be deferred (envelope-identity
#: ratchet); hi is the largest locale by UTF-8 bytes and measured 77,139 B.
#:
#: NOT raised for wire.errors.sdk_spec (PR #4349): its new key put hi at
#: 77,924 B, 100 B over, and the namespace meets the deferral criterion (SDK
#: refusals are finished strings rendered in the embedding process), so it was
#: deferred instead, returning hi to 72,321 B.
#:
#: NOT raised for wire.team_integrations (PR #4297): its 22 consent-card keys
#: put hi at 78,277 B, 453 B over; the card is rendered in-process into plain
#: strings, so the namespace was deferred instead, returning hi to 74,472 B.
EAGER_TOTAL_MAX_BYTES = 76 * 1024


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
    assert set(catalogues.DEFERRED_NAMESPACES) == {
        "wire.settings",
        "wire.settings.hub_defaults",
        "wire.settings.hub_org",
        "wire.settings.keymap",
        "wire.language_receipts",
        "wire.voice_availability",
        "wire.harness_import_detail",
        "wire.wakes_supervisor",
        "wire.session_status",
        "wire.console",
        "wire.memory",
        "wire.host_speed",
        "wire.guardrails",
        "wire.radient_saver",
        "wire.aida_cadence",
        "wire.notification_sound",
        "wire.exec",
        "wire.remote_compute",
        "wire.slash",
        "wire.slash.library",
        "wire.slash.notifications_page",
        "wire.tool_rows.bridge",
        "wire.sessions",
        "wire.operator_setup",
        "wire.github",
        "wire.errors.bedrock",
        "wire.login.bedrock",
        "wire.providers.reauth",
        "wire.errors.sdk_spec",
        "wire.team_integrations",
    }, (
        "The deferred set is a reviewed policy, not a place to park a large "
        "namespace. Adding one requires updating this assertion, recording its "
        "reason in DEFERRED_NAMESPACES, and proving no envelope identity uses "
        "it (tests/unit/i18n/test_deferred_references.py ratchets that proof)."
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


def _total_ceiling_violations() -> list[str]:
    """Eager totals over the ceiling, one line per locale (the guard's payload)."""
    violations = []
    for locale in resolve.shipped_locales():
        total = sum(
            len(catalogues.catalogue_bytes(locale, namespace))
            for namespace in _eager_namespaces(locale)
        )
        if total > EAGER_TOTAL_MAX_BYTES:
            violations.append(f"{locale} is {total} B (ceiling {EAGER_TOTAL_MAX_BYTES} B)")
    return violations


_TOTAL_FAILURE = (
    "The eager wire payload for a locale grew past its total ceiling; the "
    "per-namespace ceiling is not enough when many small files accumulate: "
)


def test_the_eager_wire_payload_stays_within_the_total_ceiling() -> None:
    violations = _total_ceiling_violations()
    assert not violations, _TOTAL_FAILURE + "; ".join(violations)


def _write_catalogue(path: Path, namespace: str, size: int) -> None:
    """Write a valid catalogue file of EXACTLY ``size`` bytes."""
    prefix = f'{{"{namespace}.key": "'.encode()
    body = size - len(prefix) - len(b'"}')
    path.write_bytes(prefix + b"a" * body + b'"}')


def test_the_total_ceiling_still_catches_an_oversized_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PROOF that the total ceiling bites, on data that is not today's tree.

    Every other test here passes on the current catalogues by construction: if
    the summation were broken (an empty locale iterator, a namespace filter the
    deferred set swallowed) the shipped catalogues would still be under the cap
    and nothing would show it. This drives the SAME check over a synthetic
    locale built to fail it — three files each well under the per-namespace
    ceiling, whose total crosses the total ceiling — and pins the exact
    boundary: at the ceiling no violation, one byte over one violation naming
    the size.
    """
    root = tmp_path / "catalogues"
    (root / "xx").mkdir(parents=True)
    monkeypatch.setattr(catalogues, "_CATALOGUES", root)
    monkeypatch.setattr(resolve, "shipped_locales", lambda: ("xx",))

    # Three small files, none near the per-file ceiling: the accumulation class
    # the per-namespace ceiling cannot see (a single over-total file would also
    # trip the per-file ceiling, which would prove the wrong guard).
    # The last file takes the remainder, so the sum sits one byte under ANY
    # ceiling (at 64 KiB, 3 * (ceiling // 3) happened to be ceiling - 1, because
    # 65,536 is 1 mod 3; 80 KiB is not, so the remainder is computed).
    third = EAGER_TOTAL_MAX_BYTES // 3
    sizes = [third, third, EAGER_TOTAL_MAX_BYTES - 1 - 2 * third]
    paths = []
    for index, size in enumerate(sizes):
        namespace = f"wire.part{index}"
        path = root / "xx" / f"{namespace}.json"
        _write_catalogue(path, namespace, size)
        assert len(catalogues.catalogue_bytes("xx", namespace)) == size
        paths.append((path, namespace, size))
    assert sum(sizes) == EAGER_TOTAL_MAX_BYTES - 1
    assert max(sizes) <= EAGER_MAX_BYTES

    assert _total_ceiling_violations() == []

    # Exactly at the ceiling: still no violation (the comparison is strict).
    path, namespace, size = paths[2]
    _write_catalogue(path, namespace, size + 1)
    assert len(catalogues.catalogue_bytes("xx", namespace)) == size + 1
    assert _total_ceiling_violations() == []

    # One byte over: reported, with the size the summation actually produced.
    _write_catalogue(path, namespace, size + 2)
    assert _total_ceiling_violations() == [
        f"xx is {EAGER_TOTAL_MAX_BYTES + 1} B (ceiling {EAGER_TOTAL_MAX_BYTES} B)"
    ]
