"""Synthetic non-English catalogues for tests of NEWLY extracted English keys.

WHY THIS EXISTS. A key is authored in English first; the translators and
independent auditors land its fr/es/... rows in a later, separate wave, so a test
written with the extraction cannot read a shipped French row that does not exist
yet. Proving the call site renders through the catalogue (and no longer prints
its old English literal) needs a locale whose text differs from English, so the
test supplies a tiny hand-written one.

The French here is a TEST FIXTURE, not product copy: it never ships, is never a
substitute for the audited translation, and is patched in at the catalogue
loader so the real render path (``messages.render`` -> ``text.tr`` resolvers)
is what the assertion exercises. Once real rows land the tests keep passing
because the patch wins over the shipped file.
"""

from __future__ import annotations

import pytest

from local_operator.i18n import catalogues


def install(
    monkeypatch: pytest.MonkeyPatch,
    locale: str,
    namespace: str,
    rows: dict[str, str],
) -> None:
    """Make ``load_catalogue(locale, namespace)`` return ``rows`` (others untouched)."""
    real = catalogues.load_catalogue

    def load(requested_locale: str, requested_namespace: str) -> dict[str, str]:
        if (requested_locale, requested_namespace) == (locale, namespace):
            return dict(rows)
        return real(requested_locale, requested_namespace)

    monkeypatch.setattr(catalogues, "load_catalogue", load)
