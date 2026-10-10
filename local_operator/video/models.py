"""The vendor family table: one request mapping per model family, for both rungs.

Every video endpoint the hub and the direct FAL rung serve speaks FAL's own
request schema — and those schemas disagree with each other in ways no
caller can discover from the API (documented in the hub's RFC: the
"per-endpoint FAL inconsistencies are the public contract, ``"8s"`` vs ``5``,
``"4k"`` vs ``"4K"``" — ``docs/MEDIA-PROVIDERS.md`` C4). This module is the
harness's half of that contract: the tool's NEUTRAL values (``duration_s: int``,
``resolution: "480p"|"720p"|"1080p"|"4k"``, ``aspect_ratio``, ``audio: bool``)
map onto each family's wire here, in ONE table, so the hub rung and the FAL
rung cannot drift and a new family is one entry.

**Grounding.** Every mapping below was read from the vendor's own live
``llms.txt`` / OpenAPI schema on 2026-10-10 (captures in the video phase's
scratchpad, ``media-video/captures/``), and the per-second rates come from the
phase's currency memos (post-promo only — manager ruling 2: a promo rate is
never banked). Where a family's field is NOT grounded, the table omits it —
the rung then sends nothing and the vendor default applies, which fails
forward at worst; a guessed field name would be a silent wrong call.

**What a family NEITHER carries nor needs:** credentials (resolved at call
time), endpoints (the model id is the endpoint), poll policy (rung-level), and
pricing *charged* by the provider (only the pre-spend estimate lives here; the
figure that gets reported at settle rides the outcome, never this table).

A value the family cannot represent (``duration_s=5`` on Veo's 4/6/8 enum, a
``4k`` request on a 480p/768p/1080p schema) is REJECTED with
:class:`VideoParamRejected`; the rung records that rung as a skipped attempt
and the walk fails forward — never a silent nearest-value substitution,
because the substitution changes what the user paid for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

__all__ = [
    "FAL_DEFAULT_MODELS",
    "GENERIC_FAMILY",
    "RADIENT_IMAGE_SOURCE_FIELD",
    "RADIENT_VIDEO_SOURCE_FIELD",
    "VIDEO_FAMILIES",
    "VIDEO_MODE_SOURCE_KIND",
    "VideoFamily",
    "VideoParamRejected",
    "estimate_unit_price_usd",
    "family_for_model",
    "map_family_params",
    "resolve_mode",
]

#: Neutral mode → the source MEDIA kind a request in that mode carries (the
#: contract's ``source_kinds`` capability field reads this). ``t2v`` has no
#: source; ``i2v`` one image; ``v2v`` one video.
VIDEO_MODE_SOURCE_KIND: Mapping[str, str | None] = {
    "t2v": None,
    "i2v": "image",
    "v2v": "video",
}

#: The hub's request field names for sources ONCE the i2i lane's enabling PR
#: ships the server-side mapping (manager note, 2026-10-10: the canonical
#: input word for video sources is ``video_url``; the image lane's edit brief
#: froze ``source_url`` for image sources). The harness rungs do not send
#: sources to the hub in v1 — ``VIDEO_RUNG_SPECS[RADIENT].sources`` is NONE
#: until that PR lands — and these constants exist so the flip is one edit
#: beside the spec.
RADIENT_IMAGE_SOURCE_FIELD = "source_url"
RADIENT_VIDEO_SOURCE_FIELD = "video_url"


class VideoParamRejected(ValueError):
    """A requested value this family's schema cannot represent.

    Raised by :func:`map_family_params`; the rung converts it to a
    ``RungSkipped``/``unsupported`` attempt so the walk fails forward to a
    family that CAN serve the value, and a request no family can serve ends
    as the honest per-rung refusal list.
    """


@dataclass(frozen=True)
class VideoFamily:
    """One model family's wire mapping (see the module docstring)."""

    name: str
    #: Model-id prefixes that select this family (matched longest-prefix-first
    #: by :func:`family_for_model`). Prefixes, not full ids, because the id
    #: suffix carries the mode/version (``.../text-to-video``,
    #: ``.../image-to-video``).
    prefixes: tuple[str, ...]
    #: Modes this family serves (subset of t2v/i2v/v2v).
    modes: frozenset[str]
    #: Duration shapes, exactly one of which is set: a discrete second set
    #: whose wire spelling is ``"<n>s"`` (Veo), an inclusive integer range
    #: whose wire spelling is a bare number, or an inclusive float range.
    duration_seconds: frozenset[int] | None = None
    duration_range: tuple[int, int] | None = None
    duration_float_range: tuple[float, float] | None = None
    #: Neutral resolution → wire spelling. ``None`` = the schema documents no
    #: resolution parameter; an explicit request against it is rejected.
    resolution: Mapping[str, str] | None = None
    #: Neutral aspects this family accepts; ``None`` = no aspect parameter.
    aspect: frozenset[str] | None = None
    #: Field name a neutral ``audio: bool`` rides (``generate_audio`` on Veo,
    #: ``audio`` on Wan). ``None`` = the schema distinguishes no such switch.
    audio_field: str | None = None
    #: Optional value mapping for ``audio_field`` (a family whose audio switch
    #: is an enum rather than a bool).
    audio_values: Mapping[bool, Any] | None = None
    #: Source field per source kind (image → i2v, video → v2v). Missing key =
    #: this family's mode does not take that source.
    source_fields: Mapping[str, str] = field(default_factory=dict)
    #: Fixed params always sent with a request (a vendor quirk that needs one
    #: deterministic value; documented at the entry).
    fixed: Mapping[str, Any] = field(default_factory=dict)
    #: Post-promo per-second US$ by neutral resolution; ``None`` = no figure
    #: (documentation-only family). Used SOLELY for the approval copy's
    #: pre-spend estimate — never written as a charged cost.
    rates_per_second: Mapping[str, float] | None = None


#: The table. Order is not lookup order (prefix length is); grouped by family
#: for review.
VIDEO_FAMILIES: tuple[VideoFamily, ...] = (
    VideoFamily(
        name="veo3.1",
        prefixes=("fal-ai/veo3.1",),
        modes=frozenset({"t2v", "i2v"}),
        # "DurationEnum: Options "4s", "6s", "8s"" — veo3.1 i2v llms.txt,
        # read 2026-10-10. Integer 8 is REJECTED by the hub's quote (C4) and
        # the vendor schema alike; the wire is the suffixed string.
        duration_seconds=frozenset({4, 6, 8}),
        # "720p", "1080p", "4k" — same source. ("4k" lowercase on Veo.)
        resolution={"720p": "720p", "1080p": "1080p", "4k": "4k"},
        aspect=frozenset({"16:9", "9:16"}),
        audio_field="generate_audio",
        source_fields={"image": "image_url"},
        # Rates (memo §2, read 2026-10-10): $0.20/s no audio, $0.40/s with at
        # 720p/1080p; 4k $0.40 no audio. The estimate quotes the NO-AUDIO
        # figure (conservative-low against the vendor's audio-on default,
        # which the approval copy states); a promo never appears here.
        rates_per_second={"720p": 0.20, "1080p": 0.20, "4k": 0.40},
    ),
    VideoFamily(
        name="wan-3.0",
        prefixes=("alibaba/wan-3.0", "fal-ai/wan/v2.7"),
        modes=frozenset({"t2v", "i2v", "v2v"}),
        # Wan 3.0/Prime t2v+i2v: integer seconds, 2-30 ("duration (5, nullable)",
        # hub MEDIA.md table; prime i2v "duration (integer, optional)"). The
        # v2.7 edit app's own enum is 0|2..10 where 0 is a "match input video
        # duration" sentinel — so omitting duration_s stays the match-input
        # path there, and a >10 s request on THAT app fails forward at the
        # vendor; the family range quotes the wider Prime schema it shares
        # its prefix space with.
        duration_range=(2, 30),
        resolution={"480p": "480p", "720p": "720p", "1080p": "1080p"},
        aspect=frozenset({"16:9", "9:16", "1:1"}),
        audio_field="audio",
        source_fields={"image": "start_image_url", "video": "video_url"},
        # Wan 3.0 (non-prime) post-promo memo rates $0.05/$0.10/$0.20; the
        # figure here is the CHEAPEST member of the family the estimate may
        # name (prime quotes $0.068/$0.14/$0.28 on the hub's own price row at
        # quote time; this table only serves the pre-spend APPROXIMATION, and
        # the approval copy says "~").
        rates_per_second={"480p": 0.05, "720p": 0.10, "1080p": 0.20},
    ),
    VideoFamily(
        name="gemini-omni-flash",
        prefixes=("google/gemini-omni-flash",),
        modes=frozenset({"t2v", "i2v"}),
        # "duration (integer): Range: 3 to 10" — v1.1 t2v capture, read
        # 2026-10-10.
        duration_range=(3, 10),
        resolution={"360p": "360p", "720p": "720p", "1080p": "1080p", "4k": "4k"},
        aspect=frozenset({"16:9", "9:16"}),
        source_fields={"image": "image_url"},
        # "360p $0.03, 720p $0.10, 1080p $0.15, 4K $0.30" — llms.txt.
        rates_per_second={"360p": 0.03, "720p": 0.10, "1080p": 0.15, "4k": 0.30},
    ),
    VideoFamily(
        name="h3-max",
        prefixes=("minimax/h3-max",),
        modes=frozenset({"t2v", "i2v"}),
        # "duration (float): Range: 0.92 to 15" — h3-max i2v capture. ints
        # ride as JSON numbers on a float schema.
        duration_float_range=(0.92, 15.0),
        # "480P", "768P", "1080P" — CAPITAL P on this family; the hub's C4
        # note names exactly this kind of per-endpoint casing as frozen
        # public contract. A neutral 720p request maps to the family's 768P
        # ladder step (the vendor's own naming for that tier); a neutral 4k
        # request is REJECTED (not in the schema).
        resolution={"480p": "480P", "720p": "768P", "1080p": "1080P"},
        source_fields={"image": "image_url"},
        # `prompt_expansion_mode` is REQUIRED in the schema and its own
        # example sends "disabled"; the agent writes the prompt deliberately,
        # so the harness pins the deterministic value rather than spending
        # up to ~30 s (and drifting the prompt) on 'balanced' — the schema's
        # default.
        fixed={"prompt_expansion_mode": "disabled"},
        # Post-promo memo rates: $0.05/$0.08/$0.16 (480/768/1080). The
        # Oct-15 promo cliff is never banked (ruling 2).
        rates_per_second={"480p": 0.05, "720p": 0.08, "1080p": 0.16},
    ),
    # NOTE: the kling families' endpoint slugs are view-redacted in every
    # capture this phase holds, and the phase's top rule forbids transcribing
    # one from a view — so there is intentionally NO kling entry here. Kling
    # models resolve to the GENERIC mapping below (numeric duration, sources
    # only) until a kling pin moves from a capture into this table in the
    # same change that quotes it.
)

#: The GENERIC mapping for a model family no entry names. Deliberately
#: narrow: duration as an integer (the majority convention) and NO
#: resolution/audio/aspect — a field the family did not document is omitted
#: rather than guessed past the schema, and the vendor's own rejection
#: (surfaced as a failed attempt) is the honest answer for anything narrower.
#: Registered as a family so callers keep one lookup path.
GENERIC_FAMILY = VideoFamily(
    name="generic",
    prefixes=(),
    modes=frozenset({"t2v", "i2v", "v2v"}),
    duration_range=(1, 60),  # the hub's platform quote bound (maxQuotedDurationSeconds)
    source_fields={"image": "image_url", "video": "video_url"},
)


def family_for_model(model_id: str) -> VideoFamily:
    """The family for ``model_id``; the generic mapping when nothing matches.

    Longest-prefix-first so ``minimax/h3-max-turbo`` selects the h3-max family
    over a shorter sibling prefix.
    """
    candidate = (model_id or "").strip()
    best: VideoFamily | None = None
    best_len = -1
    for family in VIDEO_FAMILIES:
        for prefix in family.prefixes:
            if candidate.startswith(prefix) and len(prefix) > best_len:
                best = family
                best_len = len(prefix)
    return best if best is not None else GENERIC_FAMILY


def resolve_mode(*, source_image: bool, source_video: bool) -> str:
    """The mode a request's sources imply (the tool's derived-mode rule).

    Exactly one source total is the tool's contract; this helper is the ONE
    spelling of "video source → v2v; image source → i2v; none → t2v" so the
    tool and the runner cannot disagree about the mode a job started in.
    """
    if source_video:
        return "v2v"
    if source_image:
        return "i2v"
    return "t2v"


def _duration_wire(family: VideoFamily, seconds: int) -> Any:
    if family.duration_seconds is not None:
        if seconds in family.duration_seconds:
            return f"{seconds}s"
        allowed = ", ".join(f"{value}s" for value in sorted(family.duration_seconds))
        raise VideoParamRejected(f"{family.name} supports only {allowed} — got {seconds}s.")
    if family.duration_range is not None:
        low, high = family.duration_range
        if low <= seconds <= high:
            return seconds
        raise VideoParamRejected(
            f"{family.name} supports durations {low}-{high}s — got {seconds}s."
        )
    if family.duration_float_range is not None:
        low, high = family.duration_float_range
        if low <= seconds <= high:
            return seconds
        raise VideoParamRejected(
            f"{family.name} supports durations {low:g}-{high:g}s — got {seconds}s."
        )
    raise VideoParamRejected(f"{family.name} documents no duration parameter.")


def map_family_params(
    family: VideoFamily,
    *,
    prompt: str,
    duration_s: int | None,
    resolution: str | None,
    aspect_ratio: str | None,
    audio: bool | None,
    seed: int | None,
) -> dict[str, Any]:
    """Neutral params → the family's wire params; ``VideoParamRejected`` on a
    value the family cannot represent.

    Omission means "the vendor's default applies" (the tool's ``None``
    contract); this function never invents a value for an omitted parameter.
    """
    params: dict[str, Any] = {"prompt": prompt}
    params.update(dict(family.fixed))
    if duration_s is not None:
        params["duration"] = _duration_wire(family, duration_s)
    if resolution is not None:
        if family.resolution is None:
            raise VideoParamRejected(f"{family.name} documents no resolution parameter.")
        mapped = family.resolution.get(resolution)
        if mapped is None:
            supported = ", ".join(sorted(family.resolution))
            raise VideoParamRejected(
                f"{family.name} supports resolutions {supported} — got {resolution}."
            )
        params["resolution"] = mapped
    if aspect_ratio is not None:
        if family.aspect is None or aspect_ratio not in family.aspect:
            raise VideoParamRejected(f"{family.name} does not offer aspect ratio {aspect_ratio}.")
        params["aspect_ratio"] = aspect_ratio
    if audio is not None:
        if family.audio_field is None:
            raise VideoParamRejected(f"{family.name} documents no audio switch.")
        if family.audio_values is not None:
            params[family.audio_field] = family.audio_values[audio]
        else:
            params[family.audio_field] = audio
    if seed is not None:
        params["seed"] = seed
    return params


def estimate_unit_price_usd(family: VideoFamily, resolution: str | None) -> float | None:
    """Post-promo per-second estimate for the approval copy, or ``None``.

    Documentation-grade arithmetic ONLY: the approval prompt needs a pre-spend
    figure, the settle-time receipt carries whatever the provider REPORTS, and
    the two must not be confused (design D8). ``resolution=None`` quotes the
    family's lowest listed rate — the cheap tier a default run approaches —
    because the prompt must not silently overstate the spend either.
    """
    rates = family.rates_per_second
    if not rates:
        return None
    if resolution is not None:
        return rates.get(resolution)
    return min(rates.values())


#: The FAL rung's default model per mode when the caller pins none. Chosen
#: from the phase's currency memos (read 2026-10-10), NEVER a veo3.1 pin
#: (ruling 1: no new veo pins; the family retires). ``t2v``: Wan 3.0 — the
#: Artificial Analysis T2V #1 at $0.10/s 720p. ``i2v``: h3-max — AA I2V #1,
#: post-promo $0.08/s at its 768P tier. ``v2v``: wan v2.7 edit — the pin
#: set's first entry with an id visible in every capture. These are defaults,
#: not pins: any id the caller passes wins, and the hub rung has no table
#: like this (its default is the hub's own ``default: true`` row).
FAL_DEFAULT_MODELS: Mapping[str, str] = {
    "t2v": "alibaba/wan-3.0/text-to-video",
    "i2v": "minimax/h3-max/image-to-video",
    "v2v": "fal-ai/wan/v2.7/edit-video",
}
