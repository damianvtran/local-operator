"""Display names for models, and the honesty rule that bounds them.

WHY THIS EXISTS
---------------
The status band rendered a model as its SELECTOR — ``anthropic/claude-opus-5``,
23 cells — and the left group has to hold the model, the reasoning effort and
the working directory in the same run. Measured on the shipped registry, the
selector is the widest way to say every one of these:

====================================== ===== ============================ =====
selector                               cells display name                cells
====================================== ===== ============================ =====
``anthropic/claude-opus-5``               23 ``Claude Opus 5``               13
``anthropic/claude-sonnet-4-6``           26 ``Claude Sonnet 4.6``           17
``anthropic/claude-opus-4-5-20251101``    36 ``Claude Opus 4.5``             15
``alibaba/qwen2.5-coder-32b-instruct``    36 ``Qwen 2.5 Coder 32B
                                             Instruct``                      27
====================================== ===== ============================ =====

Nothing new has to be fetched to get the right-hand column. ``ModelInfo.name``
has always carried it, from both sources the owner asked about: the curated
registry rows ship one, and :func:`local_operator.model.discovery.merge_models`
already layers a provider's ``/v1/models`` name over the curated one for the
model picker (``CatalogueEntry.label``). What was missing is not a data source.
It is a policy for when a short name is SAFE, because a short name that names
two models is worse than a long one that is merely wide.

PRECEDENCE
----------
1. ``name`` handed in by the caller. This is ``ModelInfo.name`` as the caller
   already resolved it — ``resolve_model_info`` for a running session's spec,
   ``merge_models`` for a picker row — so the "curated or provider endpoint?"
   question is answered exactly once, in the layer that owns the merge, and this
   module never re-litigates it. A blank name, or one that merely echoes the id
   back (the very common shape of a lean OpenAI-compatible gateway), counts as
   no name at all; that is the rule ``discovery._merge_name`` already documents.
2. The curated ``ModelInfo.name`` for this provider and id. Reached when the
   caller has no resolved metadata — the band paints before any listing
   resolves, and it repaints eight times a second, so its default path must be
   in-memory and free.
3. Nothing. The selector is kept VERBATIM, which is what the band rendered
   before this module existed. A local Ollama tag and a brand-new aggregator id
   nobody has curated both land here, and both are better served by the string
   the operator typed than by an invented abbreviation of it.

THE HONESTY RULE
----------------
A name is used only when it identifies exactly ONE model. Two independent things
enforce that, because two different populations break it in different ways.

**Curated names are checked against an index of every curated name in the
shipped registry** — every hosting :data:`STATIC_MODEL_HOSTINGS` names, which is
the eight direct providers. The index is deliberately built from the SHIPPED
rows and nothing else, so the answer is deterministic: a band whose text changes
because a background listing fetch landed is its own defect. A name absent from
the index is free. Two outcomes:

* the name is shared by two curated models of one provider — refused;
* the name is a curated name belonging to a DIFFERENT provider — refused, so a
  route that borrows a direct provider's marketing string cannot render
  identically to the direct route.

**A RESELLER's listing name is never used at all**, whatever it says. This is
not conservatism, it is arithmetic: a reseller's name describes the MODEL, and
every reseller resells the same models, so such a name cannot say which route is
answering. Measured against this machine's real cached listings —
``~/.local-operator/cache/{openrouter,radient}.listing.json``, 401 and 400 rows
— **398 names are carried by a row in BOTH catalogues** (``Meta: Muse Spark
1.2``, ``Qwen: Qwen3.8 Max``, ``DeepSeek V4 Flash Latest``, …) while zero names
collide *within* either one. So the collision is not a corner case for resold
models, it is the norm, and no index built from shipped data can see it.

That matters more for a resold model than anywhere else: price, quota and rate
limit differ between routes, and choosing the route is the whole reason a user
went through an aggregator. So a resold model keeps its selector — provider
first, unique by construction, and exactly what the band showed before this
module existed. The improvement is spent where it is safe: a DIRECT provider
whose new release the registry has not been taught about yet still resolves its
name from that provider's own listing (``anthropic/claude-opus-6`` →
``Claude Opus 6``), which is the case ``ModelSpec.display_name`` exists for and
the one the shipped registry provably lags.

Rejection is not a failure mode. It degrades to the selector, which is unique by
construction. That is the trade the whole module is built around: shorter when
short is unambiguous, wide when it is not.

WHAT THE COMPACT FORM DOES NOT FIX
----------------------------------
Under width pressure the band asks for :attr:`ModelLabel.compact`, which for a
refused name is the bare model id — so a resold ``claude-opus-5`` and the direct
``Claude Opus 5`` both land near 13 cells and differ only in case and hyphens.
That is a deliberate, recorded decision rather than an oversight. Keeping the
provider there instead (``openrouter/claude-opus-5``, 24 cells) was rendered and
measured: it costs the effort segment at 80 columns, the cost reading at 70 and
the MCP indicator at 60, which breaks the width budget this segment is under. It
is also a strict improvement on the behaviour it replaced, where BOTH routes
rendered the identical string ``claude-opus-5``. The compact rung has always
traded precision for cells; the full rung is where the guarantee lives.

WHY NO PROVIDER GLYPH
---------------------
The reference frame this was modelled on carries a provider mark beside the
name. It was considered and rejected. The band already spends a cell and a space
on ``◆`` for this segment, so a provider mark is a SECOND glyph — 2 cells, not
one — and it buys an unlearnable code with no legend anywhere in the band. It
also still cannot separate two aggregators reselling one model, which is the
exact case that motivates keeping the provider at all; the rules above give the
stronger guarantee for no cells and no new vocabulary. Where they cannot be met,
the provider comes back as a WORD, inside the selector, which needs no legend.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass
from typing import Callable, Mapping

from rich.cells import cell_len

from local_operator.model.registry import STATIC_MODEL_HOSTINGS, static_models

#: A trailing parenthesised qualifier: the release date or channel a curated
#: name carries to separate two snapshots of one model — ``Claude Opus 4.5
#: (2025-11-01)``, ``Claude 3.7 Sonnet (Latest)``. Dropping it is the only
#: shortening this module performs, and it is safe precisely because it is
#: checked for ambiguity afterwards like everything else here. Non-nested by
#: design: ``[^()]*`` cannot run past an inner bracket into a name it should
#: have left alone.
_QUALIFIER = re.compile(r"\s*\([^()]*\)$")

#: How much narrower the bare model id has to be before the compact form gives
#: up a human name for it. NOT zero, which is what this was: the clamp then
#: fired on a one-cell saving and turned ``Qwen 2.5 Coder 32B Instruct`` into
#: ``qwen2.5-coder-32b-instruct`` — a full change of identity, title case and
#: spaces becoming lowercase and hyphens, bought for a single column. Measured
#: over the shipped registry, a zero margin discards a name on 11 rows and seven
#: of those save one or two cells (the six ``Qwen 2.5 Coder`` rows save exactly
#: one; ``Claude 3.7 Sonnet (Latest)`` saves two). At 4 those seven keep their
#: names while the savings worth having still clamp — ``OpenAI o3`` → ``o3`` (7),
#: ``OpenAI o4 mini`` → ``o4-mini`` (7), ``DeepSeek V4 Flash (2026-07-31)`` →
#: ``deepseek-v4-flash-0731`` (8). Rendered at every width from 120 down to 50,
#: the margin costs ZERO segments: the worst case is two extra cells on one row.
_ID_MARGIN = 4


@dataclass(frozen=True)
class ModelLabel:
    """The two forms the status band needs, resolved together.

    Both are produced in one pass because the compact form's ambiguity check
    depends on the full form's outcome: a name that was rejected outright has no
    compact form to speak of, and a name whose qualifier cannot be dropped
    without colliding compacts to itself. Handing callers a single string and a
    ``short=`` flag would either recompute the whole resolution per repaint or
    leave the two forms free to disagree.
    """

    #: The widest honest rendering: a display name, or the selector.
    full: str
    #: What the band's ``shorten-model`` rung falls back to. Equal to
    #: :attr:`full` whenever there is nothing safe to drop.
    compact: str


def model_label(provider: str, model_id: str, name: str = "") -> ModelLabel:
    """Both display forms for one model.

    ``name`` is the caller's already-resolved ``ModelInfo.name`` (empty when it
    has none). ``provider``/``model_id`` are the selector's two halves, so an
    aggregator's vendor-scoped id arrives whole: ``("openrouter",
    "moonshotai/kimi-k2")``.
    """
    return _model_label(provider, model_id, name.strip())


@functools.lru_cache(maxsize=256)
def _model_label(provider: str, model_id: str, name: str) -> ModelLabel:
    """Memoized body. Keyed on the pre-stripped name so equal inputs share an
    entry, and bounded because ids reach here from user input — a typo per
    ``/model`` attempt must not grow the map without limit. Pure: the index it
    reads is the shipped registry, so nothing here can go stale.
    """
    selector = f"{provider}/{model_id}" if model_id else provider
    # What the band's ``shorten-model`` rung showed before this module existed,
    # and still the floor it may never do much worse than — see _ID_MARGIN.
    bare_id = selector.rpartition("/")[2] or selector
    chosen = _unambiguous_name(provider, model_id, selector, name)
    if not chosen:
        # No name anyone can vouch for: exactly the behaviour this segment had
        # before, selector and all.
        return ModelLabel(full=selector, compact=bare_id)
    compact = _drop_qualifier(chosen)
    if compact != chosen and not _names_one(_compact_index(), compact, selector):
        # Two snapshots of one model share everything but the qualifier, so
        # dropping it is exactly what makes them indistinguishable.
        compact = chosen
    if cell_len(bare_id) + _ID_MARGIN <= cell_len(compact):
        # A DISPLAY NAME CAN BE WIDER THAN THE ID IT REPLACES, and the rung that
        # asks for this form is the one the band reaches under width pressure —
        # so it must not be the reason another segment is dropped.
        compact = bare_id
    return ModelLabel(full=chosen, compact=compact)


def _unambiguous_name(provider: str, model_id: str, selector: str, name: str) -> str:
    """The first candidate name that names this model and no other, else "".

    The curated name is tried even after a supplied one was rejected. That is
    deliberate rather than defensive: the rejection reason is "some other model
    already answers to this", which says nothing about whether THIS model has a
    good name of its own.
    """
    if _resells(provider):
        # A reseller's listing name describes the MODEL, and every reseller
        # resells the same models — 398 of ~400 names are shared between the two
        # shipped aggregators' real catalogues. No name from here can say which
        # route is answering, and the route is what differs in price and quota.
        return ""
    curated = static_models(provider).get(model_id)
    for candidate in (name, (curated.name if curated else "").strip()):
        if not candidate or _echoes_id(candidate, model_id):
            continue
        if _names_one(_full_index(), candidate, selector):
            return candidate
    return ""


@functools.cache
def _resells(provider: str) -> bool:
    """Whether ``provider`` is a RESELLER rather than the model's own vendor.

    Imported lazily-per-provider through a memo rather than read at module
    import: ``providers.registry`` is a heavier module than this one needs to
    pull in at import time, and the answer for a given provider never changes.
    """
    from local_operator.providers.registry import AGGREGATOR_PROVIDERS

    return provider in AGGREGATOR_PROVIDERS


def _echoes_id(name: str, model_id: str) -> bool:
    """Whether a "name" is just the id handed back.

    Endpoints that carry no display metadata answer with the key they were asked
    about, and a provider that scopes its ids by vendor echoes the scoped form
    (``moonshotai/kimi-k2``) while the band would show the bare tail. Both are
    the id wearing a name's clothes: promoting either would spend the whole
    honesty budget to render the string it started from.
    """
    return name == model_id or name == (model_id.rpartition("/")[2] or model_id)


def _drop_qualifier(name: str) -> str:
    """``Claude Opus 4.5 (2025-11-01)`` → ``Claude Opus 4.5``.

    Falls back to the input when the qualifier is the whole name, which no
    shipped row does but a provider listing is free to send.
    """
    return _QUALIFIER.sub("", name).strip() or name


def _key(name: str) -> str:
    """The spelling an index and a lookup can agree on.

    CASE-FOLDED, for the same reason ``configure._normalised_id`` folds an id: a
    display name is compared here to decide whether two models are the same
    thing, and ``claude opus 5`` is not a different model from ``Claude Opus 5``.
    Exact matching let a case variant of a curated name through the refusal —
    zero real occurrences in either cached listing today, which is why this is
    one line rather than a mechanism.
    """
    return name.casefold()


def _names_one(index: Mapping[str, frozenset[str]], name: str, selector: str) -> bool:
    """Whether ``name`` is free for ``selector`` to use.

    Absent from the index means no curated model answers to it — the normal case
    for a direct provider's listing name — and is therefore free. Present means
    it is free only if this model is its sole owner.
    """
    owners = index.get(_key(name))
    return owners is None or owners == frozenset({selector})


@functools.cache
def _full_index() -> Mapping[str, frozenset[str]]:
    """Curated display name → every selector shipping it.

    Built once because the answer cannot change inside a process, not because it
    is expensive: building BOTH indexes measures 0.070 ms, which at the
    spinner's 8 fps would be 0.06% of one core. Caching it is simply free, and a
    per-repaint rebuild would be 96 dictionary insertions with nothing to show
    for them.
    """
    return _index(lambda name: name)


@functools.cache
def _compact_index() -> Mapping[str, frozenset[str]]:
    """The same index over QUALIFIER-STRIPPED names.

    Separate from :func:`_full_index` rather than derived from it at lookup time
    because the collision it detects is between the stripped forms, which the
    full index cannot see: two names that differ only inside their brackets are
    distinct keys there and the same key here.
    """
    return _index(_drop_qualifier)


def _index(key: Callable[[str], str]) -> Mapping[str, frozenset[str]]:
    """Shared body of the two indexes; ``key`` maps a curated name to its bucket."""
    buckets: dict[str, set[str]] = {}
    for hosting in STATIC_MODEL_HOSTINGS:
        for model_id, info in static_models(hosting).items():
            name = (info.name or "").strip()
            if not name:
                continue
            buckets.setdefault(_key(key(name)), set()).add(f"{hosting}/{model_id}")
    return {name: frozenset(owners) for name, owners in buckets.items()}


__all__ = ["ModelLabel", "model_label"]
