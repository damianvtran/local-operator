"""The observe-only screener provenance block sent to the hub on ack and admit.

WHAT. agent-server PR #332 (docs/TEAM-BRIDGE.md §8.5; conversations §10)
accepts one OPTIONAL top-level ``screening_observation`` key beside the verdict
on ``POST /messaging/messages/:id/ack`` and
``POST /messaging/conversations/:cid/messages/:mid/admit``::

    {"v": 1, "provider": "anthropic", "model": "claude-sonnet-5-5",
     "selection_stage": "preferred", "client_version": "0.101.14"}

It says which screener ACTUALLY answered and whether that was a preferred
screener or the failover/default cascade (operator directive 2026-11-04: pin
the preferred screeners, but descend the cascade when they are out of quota
so Team Bridge never blocks). This module only REPORTS that cascade.

INVARIANTS (each pinned in tests/unit/team_bridge/test_screening_observation.py):

* Reporting only. Nothing here may change a screening decision, its timing, a
  retry or a release: the block is built from provenance the pipeline already
  captured, adds no request and no await, and a request without it is
  byte-identical to the pre-#332 client.
* Honest provenance. ``preferred``/``fallback`` come only from the trusted
  chain's answering hop; anything else is ``unknown``. No provenance at all
  (cache hit, replayed intent, legacy receipt) omits the block — a screener is
  never invented.
* Older hubs. A pre-#332 hub decodes the conversation admit body with
  ``DisallowUnknownFields`` (measured from agent-server at ``0d1686c3^``:
  ``readConversationBody``), so an unknown key there would turn a legal admit
  into a 400. The contract's answer is the status advertisement
  ``features.team_bridge_screening_observation: 1``: a client that has not seen
  it omits the block. The memo below is fed by status reads this process
  ALREADY makes (no extra read). The one thing read back is a 400 to a body
  that carried the block (a stale advertisement after a rollback, or a rolling
  update serving both decoders behind one origin). It latches the block OFF
  for that origin for :data:`REFUSAL_COOLDOWN_S`, outranking every fresh
  advertisement meanwhile, and an admit that drew it is sent ONCE more
  without the block (``client._send_observed``) so telemetry never costs a
  message its admission — an ack is not, because no hub's ack decoder refuses
  the block; nothing else in an ack/admit response is read.
"""

from __future__ import annotations

import functools
import logging
import threading
import time
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

from local_operator.providers.registry import credential_provider_id
from local_operator.team_bridge.types import OBSERVATION_STAGES, ScreeningObservation

logger = logging.getLogger(__name__)

#: The block's wire version (§8.5). Any other version is dropped by the hub.
OBSERVATION_VERSION = 1
#: The status feature key and the one version this client speaks.
OBSERVATION_FEATURE = "team_bridge_screening_observation"
WIRE_KEY = "screening_observation"

#: Hub base URL -> whether its LAST successful status read advertised the
#: block at :data:`OBSERVATION_VERSION`. Process memory only; it is never an
#: input to any decision except "attach a telemetry key or not". Overwritten
#: on every status read, so a hub that stops advertising stops receiving it.
_HUB_SUPPORT: dict[str, bool] = {}
#: Hub origin -> monotonic deadline before which the block stays OFF there,
#: whatever a status read advertises (review Q-2). Process memory only, like
#: the advertisement memo it masks.
_REFUSED_UNTIL: dict[str, float] = {}
_HUB_LOCK = threading.Lock()

#: How long a 400 to a block-carrying request outranks fresh advertisements.
#:
#: WHY a latch at all: the conversation lane reads status before EVERY admit
#: (``run_screening_cycle`` -> ``verified_status_capture``), so behind one
#: origin serving two hub versions (a rolling update, a partial rollback) the
#: post-#332 pod's advertisement would re-open the gate before each request,
#: and each one landing on a pre-#332 pod would 400 again.
#:
#: WHY 15 minutes: an agent-server release is an Argo sync (~3 min) plus a
#: rolling update of a couple of pods, so a version-mixed window is minutes;
#: 15 covers it with margin. Longer buys little — after a PERMANENT rollback
#: the latch simply re-arms on its next 400, and since that request is resent
#: without the block (no hold, no re-screen), each expiry costs one extra
#: request, never a paid screen. Shorter would let a slow rollout 400 every few
#: minutes. Losing telemetry for the whole window costs nothing.
#:
#: Only the cool-down releases the latch. A successful request WITHOUT the
#: block there proves nothing about the block — it is exactly what the resend
#: after the 400 is — so it must not re-open the gate.
REFUSAL_COOLDOWN_S = 15 * 60.0

#: Indirection so tests can move the clock; monotonic so a wall-clock jump can
#: neither extend nor end a latch.
_monotonic = time.monotonic


def hub_origin(base_url: str) -> str:
    """The ``scheme://netloc`` key the gate is held under (also what logs name)."""

    # Keyed by ORIGIN, not the full base: the pull lane's default client and
    # the status fetcher's owner-pinned client may spell the same hub with and
    # without its ``/v1`` path (``pull._guard_destination`` accepts that), and
    # the origin is the trust boundary those guards already use.
    parts = urlsplit(str(base_url or "").strip())
    return f"{parts.scheme}://{parts.netloc}".lower()


def note_hub_status(base_url: str, payload: Any) -> None:
    """Record whether a status payload advertises the block (§8.5)."""

    features = payload.get("features") if isinstance(payload, Mapping) else None
    version = features.get(OBSERVATION_FEATURE) if isinstance(features, Mapping) else None
    # ``type(...) is int`` so a JSON ``true`` (a bool, an int subclass) is not
    # read as version 1 of a contract it does not name.
    supported = type(version) is int and version == OBSERVATION_VERSION
    with _HUB_LOCK:
        _HUB_SUPPORT[hub_origin(base_url)] = supported


def note_hub_refused(base_url: str) -> None:
    """A 400 answered a request that carried the block: latch it off here.

    An advertisement can go stale under a running process — a rollback below
    #332, or a rolling update serving both decoders behind one origin — and a
    pre-#332 admit decoder answers the unknown key with a 400. Clearing the
    memo alone does not hold: the conversation lane reads status before EVERY
    admit (``scheduler.run_screening_cycle`` -> ``verified_status_capture``),
    so a post-#332 pod's advertisement would re-open the gate before the next
    request and a pre-#332 pod would refuse it again (review Q-2; the round-1
    docstring wrongly said status was read only when files ride a message).
    So the refusal latches the block off for this origin until
    :data:`REFUSAL_COOLDOWN_S` elapses, and advertisements meanwhile are
    recorded but do not re-open it.

    Latching off is the safe direction whatever caused the 400: losing
    telemetry costs nothing. The request itself is resent once without the
    block by the client (``_send_observed``), so the refusal never becomes a
    hold — whose re-drive would retire and re-screen an unsent FALLBACK allow,
    a second paid call that may land on another model (review S-1/m1).
    """

    key = hub_origin(base_url)
    with _HUB_LOCK:
        _HUB_SUPPORT[key] = False
        _REFUSED_UNTIL[key] = _monotonic() + REFUSAL_COOLDOWN_S
    # One line per arm, the ORIGIN only (no body, verdict, account or model),
    # so an operator can tell telemetry to this hub is off for the cool-down
    # rather than silently missing (review S-r3-2). Every block-carrying 400
    # re-arms and logs, and each already cost a request, so the line count
    # tracks real refusals rather than adding a noise source of its own.
    logger.info(
        "team_bridge: hub %s refused the screening observation; latched off for this origin",
        key,
    )


def hub_accepts(base_url: str) -> bool:
    """Whether THIS hub advertised the block and no refusal latch masks it."""

    key = hub_origin(base_url)
    with _HUB_LOCK:
        until = _REFUSED_UNTIL.get(key)
        if until is not None:
            if _monotonic() < until:
                return False
            del _REFUSED_UNTIL[key]
        return _HUB_SUPPORT.get(key, False)


def reset_hub_support() -> None:
    """Forget every advertisement (tests; a sign-out needs nothing else)."""

    with _HUB_LOCK:
        _HUB_SUPPORT.clear()
        _REFUSED_UNTIL.clear()


@functools.lru_cache(maxsize=1)
def _client_version() -> str:
    # Cached: the version cannot change under a running process, and the
    # lookup reads install metadata from disk — not something an ack should
    # pay per message.
    try:
        from local_operator.update import installed_version

        return str(installed_version() or "")
    except Exception:  # noqa: BLE001 - a missing version is UNKNOWN, never a failure
        return ""


def from_answered_hop(runner: Any, actual_model: str | None) -> ScreeningObservation | None:
    """The observation for a verdict a model just produced, or ``None``.

    ``actual_model`` is the serving identity the runner returned — the model
    that answered. The stage is read from the trusted chain's ANSWERING hop
    (the same provenance ``screener_preferred`` uses); a bare runner, or a hop
    whose model differs from the answer, has no trustworthy stage or provider,
    so it reports ``unknown`` with the model alone.
    """

    if not isinstance(actual_model, str) or not actual_model:
        return None
    from local_operator.team_bridge.model_boundary import ScreenChainRunner

    hop = runner.answered_hop if isinstance(runner, ScreenChainRunner) else None
    if hop is None or hop.model_id != actual_model:
        return ScreeningObservation(provider="", model=actual_model, selection_stage="unknown")
    # Stage 1 of ``resolve_screening_chain`` stamps ``preferred=True``; every
    # stage-2 hop (the receiver's default and its fallbackChains) is the
    # failover cascade.
    # The provider is folded to the one its credential is stored under, so a
    # login flavour (``anthropic-key``, ``radient-key``) reports the endpoint
    # it authenticates rather than an id the hub counts as OTHER; the client
    # owns that alias table (#343 review) and the eligibility check already
    # folds the same way. The model is never folded: it is what answered.
    return ScreeningObservation(
        provider=credential_provider_id(str(hop.provider or "")),
        model=actual_model,
        selection_stage="preferred" if hop.preferred else "fallback",
    )


def from_release_row(release: Mapping[str, Any]) -> ScreeningObservation | None:
    """The screener behind a B4b owner release, from the signed release row.

    The release receipt's own ``model`` is the owner-release marker, but a
    model DID answer this message's screening: the hold the person released.
    Its identity and stage were captured at answer time and are bound into
    the signed row. A stored ``screener_preferred = 0`` is NOT read as
    ``fallback`` — the column defaults to 0 on legacy rows — so anything other
    than a trusted preferred answer is ``unknown``.
    """

    from local_operator.team_bridge import hold_release

    model = str(release.get("screener_model") or "")
    if not model:
        return None
    provider = str(release.get("screener_provider") or "")
    preferred = hold_release.preferred_answer(
        provider, model, release.get("screener_preferred") == 1
    )
    return ScreeningObservation(
        # Folded for the hub only, as in ``from_answered_hop``; the signed row
        # and the preferred check above keep the stored literal.
        provider=credential_provider_id(provider),
        model=model,
        selection_stage="preferred" if preferred else "unknown",
    )


def wire_block(observation: ScreeningObservation | None) -> dict[str, Any] | None:
    """The §8.5 JSON object, or ``None`` when there is nothing honest to send."""

    if observation is None or not observation.model:
        return None
    stage = observation.selection_stage
    block: dict[str, Any] = {
        "v": OBSERVATION_VERSION,
        "model": observation.model,
        "selection_stage": stage if stage in OBSERVATION_STAGES else "unknown",
    }
    if observation.provider:
        block["provider"] = observation.provider
    version = _client_version()
    if version:
        block["client_version"] = version
    return block
