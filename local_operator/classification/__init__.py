"""Classification layer: a swappable decision-model vendor for resource recommendations.

The interface contract is ``docs/design/classification-layer.md`` — §4 for this
package's surface, §5 for the context budget, §6 for the security rails, §8 for
the ``values.classification.*`` keys. This module is the ONLY public surface: the
submodules are importable, but a caller should be able to do everything it needs
through the names re-exported here.

WHAT THIS LAYER IS
==================

Once per user message the harness asks a *decision model* — a model that answers
typed questions rather than generating prose — which already-installed skills,
guides and MCP servers might help. The answer is rendered as a short advisory
block and appended to the prompt.

WHAT IT IS NOT (AND MAY NEVER BECOME)
=====================================

* **Advisory.** Nothing here may gate a capability, change an approval tier or
  alter a tool's availability (§6). A caller that acts on a recommendation
  without checking the resource itself is using this layer wrongly.
* **A replacement for the existing heuristics.** Skill/guide selection keeps
  working unchanged; this runs beside it, and the block is additive (§12).
* **Transcript-reading.** Only the newest user message, an optional short
  pre-redacted context string, and the candidate roster ever leave the process
  (§5). No history, no compaction summary, no tool results.
* **Able to fail a turn.** Every failure mode returns an empty
  :class:`Recommendation` with ``skipped`` set; with the layer disabled or the
  vendor down, the prompt is byte-identical to a harness without it (§7).

TYPICAL USE
===========

.. code-block:: python

    service = ClassificationService(manager=credentials, settings=config.values)
    if service.enabled:
        recommendation = await service.recommend_resources(
            RecommendationRequest(
                user_message=message,
                context=last_compaction_line,
                candidates=discovered_resources,  # harness-owned descriptions only
            )
        )
        if recommendation.block:
            prompt += "\\n\\n" + recommendation.block
"""

from __future__ import annotations

from local_operator.classification.cascade import (
    DEFAULT_MODEL,
    DEFAULT_VENDOR,
    VENDOR_CHOICES,
    VENDOR_ORDER,
    classification_section,
    leg_order,
    pinned_vendor,
    resolve_vendor,
    vendor_status,
)
from local_operator.classification.context import (
    CANDIDATE_LINE_LIMITS,
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_MAX_STATE_CHARS,
    KIND_DROP_ORDER,
    TRUNCATION_MARKER,
    Candidate,
    build_state,
    candidate_line,
    candidates_digest,
    max_candidates,
    max_state_chars,
    select_candidates,
    serialized_size,
    setting_int,
)
from local_operator.classification.recommend import (
    DEFAULT_MAX_RECOMMENDATIONS,
    NONE_OPTION,
    QUESTION_KIND_ORDER,
    QuestionPlan,
    Recommendation,
    RecommendationRequest,
    ResourceKind,
    build_questions,
    collect_resources,
    max_recommendations,
    option_id,
    render_block,
)
from local_operator.classification.service import (
    CACHE_SIZE,
    CIRCUIT_FAILURE_THRESHOLD,
    DEFAULT_AUTO,
    DEFAULT_NOTICE,
    DEFAULT_TIMEOUT_MS,
    ClassificationService,
)
from local_operator.classification.types import (
    Answer,
    DecisionRequest,
    DecisionResponse,
    DecisionSchemaError,
    DecisionVendor,
    DecisionVendorError,
    Question,
    QuestionKind,
    SkipReason,
)
from local_operator.classification.vendors import (
    OPENROUTER_ENDPOINT,
    OPENROUTER_MODEL,
    RADIENT_ENDPOINT,
    RADIENT_MODEL,
    TYPESAFE_ENDPOINT,
    TYPESAFE_MODEL,
    OpenRouterVendor,
    RadientVendor,
    TypeSafeVendor,
    questions_payload,
    request_body,
)

__all__ = [
    # -- §4: the vendor contract ------------------------------------------
    "Answer",
    "DecisionRequest",
    "DecisionResponse",
    "DecisionSchemaError",
    "DecisionVendor",
    "DecisionVendorError",
    "Question",
    "QuestionKind",
    "SkipReason",
    # -- §4: the cascade ---------------------------------------------------
    "DEFAULT_MODEL",
    "DEFAULT_VENDOR",
    "VENDOR_CHOICES",
    "VENDOR_ORDER",
    "classification_section",
    "leg_order",
    "pinned_vendor",
    "resolve_vendor",
    "vendor_status",
    # -- §4/§5: candidates and the state builder ---------------------------
    "CANDIDATE_LINE_LIMITS",
    "DEFAULT_MAX_CANDIDATES",
    "DEFAULT_MAX_STATE_CHARS",
    "KIND_DROP_ORDER",
    "TRUNCATION_MARKER",
    "Candidate",
    "build_state",
    "candidate_line",
    "candidates_digest",
    "max_candidates",
    "max_state_chars",
    "select_candidates",
    "serialized_size",
    "setting_int",
    # -- §4: the recommendation layer --------------------------------------
    "DEFAULT_MAX_RECOMMENDATIONS",
    "NONE_OPTION",
    "QUESTION_KIND_ORDER",
    "QuestionPlan",
    "Recommendation",
    "RecommendationRequest",
    "ResourceKind",
    "build_questions",
    "collect_resources",
    "max_recommendations",
    "option_id",
    "render_block",
    # -- §4: the service ---------------------------------------------------
    "CACHE_SIZE",
    "CIRCUIT_FAILURE_THRESHOLD",
    "DEFAULT_AUTO",
    "DEFAULT_NOTICE",
    "DEFAULT_TIMEOUT_MS",
    "ClassificationService",
    # -- §3: the legs themselves (tests and diagnostics) -------------------
    "OPENROUTER_ENDPOINT",
    "OPENROUTER_MODEL",
    "RADIENT_ENDPOINT",
    "RADIENT_MODEL",
    "TYPESAFE_ENDPOINT",
    "TYPESAFE_MODEL",
    "OpenRouterVendor",
    "RadientVendor",
    "TypeSafeVendor",
    "questions_payload",
    "request_body",
]
