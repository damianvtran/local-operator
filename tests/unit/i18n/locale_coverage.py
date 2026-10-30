"""Which locales carry which catalog namespace — derived, never hand-listed.

Two questions keep recurring across the i18n tests: which locales carry a given
namespace (so the English-source assertions can be aimed at the rest), and which
``(locale, namespace)`` pairs are genuinely absent (so the raw catalogue route's
404 is aimed at a real absence). Both were answered with a tuple of locale codes
in the test file, and that tuple is what goes stale: this suite's own comments
record `es`, `ru` and `vi` each being hand-moved as their catalogues landed, and
`relay.account_policy` in `ru` still reddened main on 2026-10-30. A translation
wave that is working must not turn a test red.

So the sets come from the COMMITTED CATALOGUES — the same files the emitter, the
checker and the runtime read — and the locale universe is the generated shipment
ledger (``resolve.shipped_locales()``), never a list declared here or there. A
ninth locale, or `hi`/`es` landing tomorrow, needs no edit in any caller.

**Membership is per NAMESPACE FILE, not per message id.** `catalogues.namespaces`
answers "does the locale ship this file", which is the question both callers ask:
the route serves or 404s a whole file, and a fallback assertion over a whole
namespace's keys is decided by whether the file exists. (A locale can carry a
file and still lack a specific id; `messages.render_record` then falls back per
id. Callers that need that finer question should read their own catalogue, as the
byte-fidelity assertions in these files already do.)

**The vacuity guard FAILS, and that is a decision.** "A locale without this
namespace renders the English source" is only evidence while some shipped locale
lacks the namespace; once every locale carries it, the loop compares nothing and
passes for the wrong reason. :func:`require_nonempty` therefore FAILS that half,
with the derived sets named in the message.

Failing rather than skipping is deliberate: a skip is reported once and then reads
as green on every subsequent run, which is exactly how the file this helper
replaces rotted — its locale list stayed green while the corpus moved under it. A
failure forces the loss of coverage to be dealt with. The fix it should prompt is
to pin the fallback against a synthetic catalogue root (``test_catalogues.py``
builds fixture trees that way), never to restore a locale list.

Caches are keyed by the catalogue root, so a test that swaps the tree (as
`test_catalogues.py` does with ``monkeypatch.setattr(catalogues, "_CATALOGUES")``)
is answered from that tree instead of from a stale reading of the shipped corpus.
"""

from __future__ import annotations

import functools
from collections.abc import Iterable, Sequence

from local_operator.i18n import catalogues, resolve


def targets() -> tuple[str, ...]:
    """Shipped locales other than the source — the ones `messages.render` honours.

    Unshipped locales are excluded because :func:`resolve.eligible_locale` coerces
    them to ``en``: rendering one would compare the English source with itself and
    pass for the wrong reason.
    """
    return tuple(locale for locale in resolve.shipped_locales() if locale != "en")


def _root() -> str:
    return str(catalogues.catalogue_root())


@functools.cache
def _namespaces(root: str, locale: str) -> frozenset[str]:
    """A locale's namespace files. The cache key carries the root it was read from."""
    del root  # the key, not an input: `catalogues` reads the live root itself
    try:
        return frozenset(catalogues.namespaces(locale))
    except catalogues.CatalogueNotFound:
        return frozenset()


def carries(locale: str, namespace: str) -> bool:
    """Whether the locale ships its own file for ``namespace``."""
    return namespace in _namespaces(_root(), locale)


def carrying(namespace: str) -> tuple[str, ...]:
    """Shipped locales with their own catalogue for ``namespace``, in ledger order."""
    return tuple(locale for locale in targets() if carries(locale, namespace))


def absent(namespace: str) -> tuple[str, ...]:
    """Shipped locales that must fall back to the English source for ``namespace``."""
    return tuple(locale for locale in targets() if not carries(locale, namespace))


def source_namespaces() -> tuple[str, ...]:
    """Namespaces with authored copy in ``en``, sorted.

    Empty ``en`` stubs (the extraction placeholders RFC §5 shipped) are excluded:
    there is nothing to translate or fall back to, so a locale "missing" one is not
    evidence about the fallback path. Cached per catalogue root — the corpus is
    package data and immutable between swaps, and the callers below walk it once
    per locale.
    """
    return _source_namespaces(_root())


@functools.cache
def _source_namespaces(root: str) -> tuple[str, ...]:
    del root  # the cache key, not an input: `catalogues` reads the live root
    return tuple(
        namespace
        for namespace in catalogues.namespaces("en")
        if catalogues.load_catalogue("en", namespace)
    )


def absent_pairs(locales: Iterable[str] | None = None) -> tuple[tuple[str, str], ...]:
    """``(locale, namespace)`` pairs where a shipped locale lacks an authored namespace.

    Sorted and deterministic, so a caller can take a bounded sample and still name
    the same pair between runs of the same tree.
    """
    wanted = sorted(locales) if locales is not None else list(targets())
    shipped = set(targets())
    return tuple(
        (locale, namespace)
        for locale in wanted
        if locale in shipped
        for namespace in source_namespaces()
        if not carries(locale, namespace)
    )


def require_nonempty(items: Sequence[object], *, subject: str, detail: str) -> None:
    """Fail the caller's half when the derived corpus has nothing left for it.

    ``detail`` says what state was found, and the message names the replacement the
    caller should reach for once the real corpus can no longer show the state — see
    the module docstring for why this fails rather than skips.
    """
    if items:
        return
    raise AssertionError(
        f"{subject}: nothing left to exercise on the current corpus ({detail}); "
        f"pin this contract against a synthetic catalogue root "
        f"(tests/unit/i18n/test_catalogues.py) instead of restoring a locale list"
    )


def reset() -> None:
    """Drop the cached readings after a test mutates the catalogue tree in place.

    The cache keys on the root PATH, which an added or deleted namespace file does
    not change: a caller that writes a catalogue into the shipped tree must call
    this before re-deriving, or it is answered from the tree as it was.
    """
    _namespaces.cache_clear()
    _source_namespaces.cache_clear()
