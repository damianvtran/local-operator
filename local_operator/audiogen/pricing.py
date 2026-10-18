"""The audio lane's ONE central rate table, for BYOK estimates.

**What this is for.** The server plane quotes and bills (``cost.amount_usd``
is authoritative there and is carried through untouched). The BYOK legs —
ElevenLabs and OpenAI, called directly with the user's own key — return raw
audio and no per-call figure at all (``cost: null`` by contract). The harness
therefore ESTIMATES, from the single table below, exactly as the frozen
contract's cost rules demand: labelled ``estimated`` at ``rate_table`` source
with :data:`PRICE_VERSION` naming the table — never written as a charge, never
mixed into a billed figure.

**One table, one version.** Every rate lives HERE and nowhere else. The
version string travels in the cost provenance of every estimate, so a future
price move is a new version plus new rows, and a row that cannot be priced
answers ``None`` (unknown → None — an unpriceable call is "amount not stated",
which the ledger already renders honestly, never a fabricated zero).

**Provenance and dates.** Rates below were read on 2026-10-17; each row names
its source. They are ESTIMATES for a receipt, not billing authority:
ElevenLabs bills credits against a plan, OpenAI bills tokens, and both can
change their price pages without changing this table — which is exactly why
:data:`PRICE_VERSION` exists.

**What is deliberately NOT here.** Radient rates: the platform's quote and
settled figures arrive per request on the wire, and a client-side copy of the
server's prices would be a second source of truth for money the server already
states.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "PRICE_VERSION",
    "PriceEstimate",
    "sound_effect_estimate",
    "sound_effect_units",
    "voiceover_estimate",
]

#: The table's version label — rides ``billing_basis="estimated"`` rows'
#: ``price_version``/provenance, making an estimate visible instead of hidden
#: (the design §4.3 convention the web-search table established).
PRICE_VERSION = "audio-rates@2026-10-17"

#: USD per 1,000 characters, voiceover, keyed ``(provider, model)``.
#: Sources read 2026-10-17:
#: - ElevenLabs API pricing page (elevenlabs.io/pricing/api): $0.08 / 1k
#:   chars multilingual models, $0.04 / 1k chars Flash/Turbo.
#: - OpenAI: tts-1 $15 / 1M chars, tts-1-hd $30 / 1M chars; gpt-4o-mini-tts
#:   is token-priced (~$0.015 per minute of audio ≈ $0.017 / 1k chars at a
#:   ~900 chars/min speech rate — the conversion is an approximation and is
#:   labelled as one here and in the receipt).
VOICEOVER_RATES_PER_1K_CHARS: dict[tuple[str, str], float] = {
    ("elevenlabs", "eleven_multilingual_v2"): 0.08,
    ("elevenlabs", "eleven_flash_v2_5"): 0.04,
    ("openai", "tts-1"): 0.015,
    ("openai", "tts-1-hd"): 0.030,
    ("openai", "gpt-4o-mini-tts"): 0.017,
}

#: USD per SECOND of generated audio, sound effects, keyed
#: ``(provider, model)``. ElevenLabs' pricing page (read 2026-10-17) lists
#: Sound Effects at $0.12 per minute → $0.002 per second. There is no OpenAI
#: sound-effect route (the frozen contract is voiceover only), so no OpenAI
#: row exists — deliberately absent rather than zeroed.
SOUND_EFFECT_RATES_PER_SECOND: dict[tuple[str, str], float] = {
    ("elevenlabs", "eleven_text_to_sound_v2"): 0.002,
}


@dataclass(frozen=True)
class PriceEstimate:
    """One modelled figure: the amount and exactly what it models."""

    amount_usd: float
    units: float
    unit: str
    price_version: str
    #: Human-readable source line for the receipt (doc + date + approximation
    #: caveats), never itself a price.
    provenance: str


def _source_line(provider: str, model: str) -> str:
    return (
        f"{provider} {model} at the central audio rate table ({PRICE_VERSION}); "
        "an estimate from published vendor rates, not a billed figure"
    )


def voiceover_estimate(provider: str, model: str | None, chars: int) -> PriceEstimate | None:
    """The estimated voiceover cost for ``chars`` characters, or ``None``.

    ``None`` is the honest answer for every unpriceable combination — an
    unknown provider, an unknown model, a non-positive count. Callers render
    it as "amount not stated", never as zero.
    """
    if chars <= 0:
        return None
    resolved = (model or "").strip()
    rate = VOICEOVER_RATES_PER_1K_CHARS.get((provider, resolved))
    if rate is None and not resolved:
        # An unpinned model prices on the provider's DEFAULT model — the same
        # resolution the rung applies at call time (``default_voiceover_model``).
        from local_operator.audiogen.models import default_voiceover_model

        resolved = default_voiceover_model(provider)
        rate = VOICEOVER_RATES_PER_1K_CHARS.get((provider, resolved))
    if rate is None:
        return None
    return PriceEstimate(
        amount_usd=rate * (chars / 1000.0),
        units=float(chars),
        unit="chars",
        price_version=PRICE_VERSION,
        provenance=_source_line(provider, resolved),
    )


def sound_effect_units(duration_s: float) -> int:
    """The integer count a sound effect is metered in: whole started seconds.

    The platform's own sound-effect lane charges the REQUESTED duration in
    whole seconds, rounded up (``units = ceil(requested)``): the ledger's
    count is an integer and a started second is the charged unit. So 0.5 s is
    one unit and 4.01 s is five. The rate is per second, so multiplying the
    rate by a FRACTIONAL duration under-estimates a sub-second request at half
    its true cost (the merged lane did exactly that: 0.5 s priced at $0.0010
    against the $0.0020 the server charges).

    Shared by the estimate and the channel row so the amount and the count
    cannot disagree. A non-positive duration has no units; both callers treat
    it as unpriceable before asking.
    """
    return int(math.ceil(duration_s))


def sound_effect_estimate(
    provider: str, model: str | None, duration_s: float
) -> PriceEstimate | None:
    """The estimated sound-effect cost for ``duration_s`` seconds, or ``None``.

    Priced over whole started seconds (:func:`sound_effect_units`), matching
    the ledger's integer count and the server lane's billing rule — never over
    the raw fractional duration, which would quote half the real unit cost for
    anything under a second. ``None`` stays the honest answer for an unknown
    provider/model ("amount not stated"), never a zero.
    """
    if duration_s <= 0:
        return None
    resolved = (model or "").strip() or "eleven_text_to_sound_v2"
    rate = SOUND_EFFECT_RATES_PER_SECOND.get((provider, resolved))
    if rate is None:
        return None
    units = sound_effect_units(duration_s)
    return PriceEstimate(
        amount_usd=rate * units,
        units=float(units),
        unit="seconds",
        price_version=PRICE_VERSION,
        provenance=_source_line(provider, resolved),
    )
