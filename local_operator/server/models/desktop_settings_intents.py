"""Wire models for durable desktop settings intents.

The contract is ``server/utils/settings_intents.py``'s (and the design's §2b):
one :class:`SettingsIntentSlot` per kind per session, recorded by ``POST`` and
applied in the background. These models are the ROUTE's shape only; the store
keeps plain dicts so a slot written by one build is read by the next without a
migration step.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SettingsIntentKind = Literal["model", "effort", "fast", "cwd"]
SettingsIntentState = Literal[
    "pending",
    "applying",
    "queued_next_step",
    "waiting_idle",
    "applied",
    "failed",
    "superseded",
]


class SettingsIntentRequest(BaseModel):
    """``POST /v1/desktop/sessions/{id}/settings-intents``.

    SHAPE REFUSALS ARE PYDANTIC'S (422), deliberately: a wrong type for the kind
    or an ``effort`` beside a non-model kind is a client defect, not a state a
    person can act on, so it carries the validator's developer-facing detail
    rather than catalogued copy. Model/effort availability — a person's problem —
    is checked by the route against the local catalogue and refused with keyed
    copy.
    """

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1, max_length=128)
    kind: SettingsIntentKind
    value: str | bool
    #: Pairs a reasoning level with a MODEL pick so both land in one apply.
    effort: str | None = Field(default=None, max_length=32)

    @model_validator(mode="after")
    def _shape(self) -> "SettingsIntentRequest":
        if self.kind == "fast":
            if not isinstance(self.value, bool):
                raise ValueError("fast takes a boolean value")
        else:
            if not isinstance(self.value, str) or not self.value.strip():
                raise ValueError(f"{self.kind} takes a non-empty string value")
            self.value = self.value.strip()
        if self.effort is not None:
            if self.kind != "model":
                raise ValueError("effort is accepted only with kind=model")
            self.effort = self.effort.strip().lower() or None
        if self.kind == "effort":
            self.value = str(self.value).lower()
        return self


class SettingsIntentError(BaseModel):
    model_config = ConfigDict(extra="allow")

    code: str
    message: str
    retryable: bool
    #: Only on ``retry_budget_exhausted``: the code of the last failed attempt.
    last_code: str | None = None


class SettingsIntentSlot(BaseModel):
    model_config = ConfigDict(extra="allow")

    kind: SettingsIntentKind
    value: str | bool
    effort: str | None = None
    request_id: str
    generation: int
    base: Any = None
    state: SettingsIntentState
    attempts: int = 0
    created_at: str | None = None
    updated_at: str | None = None
    next_attempt_at: str | None = None
    deadline_at: str | None = None
    error: SettingsIntentError | None = None

    @model_validator(mode="before")
    @classmethod
    def _public(cls, data: Any) -> Any:
        # ``extra="allow"`` keeps forward-compatible fields, but the store's
        # ``_``-prefixed bookkeeping (``_predecessors``, ``_paired_generation``)
        # is the applier's private memory, never wire (review n1).
        if isinstance(data, dict):
            return {key: value for key, value in data.items() if not str(key).startswith("_")}
        return data


class SettingsIntentReceipt(BaseModel):
    """``POST``'s answer: the slot just written and every slot of the session."""

    intent: SettingsIntentSlot
    intents: dict[str, SettingsIntentSlot]


class SettingsIntentIndex(BaseModel):
    """``GET /v1/desktop/settings-intents``: ``{session_id: {kind: slot}}``."""

    intents: dict[str, dict[str, SettingsIntentSlot]]
