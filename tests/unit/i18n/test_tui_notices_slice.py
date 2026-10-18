"""The `tui.notices` slice: harness notices on the catalogue.

What this file pins, and why each pin exists:

1. **Byte fidelity.** Every catalogue value reproduces the argparse-era literal
   it replaced (the one permitted transformation is ICU's doubled apostrophe),
   so the notice a user reads is unchanged in `en`.
2. **The additive event contract (RFC §2.6).** A notice now carries
   `code`/`params` beside `text`; older clients read `text` verbatim, and a
   locale-aware viewer can re-render without parsing prose.
3. **Per-namespace fallback.** The namespace is new and en-only: a shipped
   locale that does not carry it (`fr`) still reads the English text.
"""

from __future__ import annotations

from pathlib import Path

import local_operator
from local_operator.harness.loop import _notice
from local_operator.harness.types import NoticeEvent
from local_operator.i18n import catalogues, messages
from local_operator.i18n.keys import tui_notices as keys

NAMESPACE = "tui.notices"
_ROOT = Path(local_operator.__file__).resolve().parent.parent

#: The exact sentence each key replaced, after ICU's one permitted
#: transformation (`''` -> `'`). Written out here because the literals no
#: longer exist in the loop: this mapping IS the byte-fidelity record.
REPLACED_LITERALS = {
    "reasoning_echo_filled": (
        "the provider refused this request for a missing reasoning echo — retrying with "
        "the echo filled in at the same effort"
    ),
    "reasoning_echo_thinking_off": (
        "the provider refused this request for a missing reasoning echo — retrying with "
        "thinking disabled"
    ),
    "budget_exhausted_retry": (
        "model spent its whole output budget thinking and produced nothing — retrying once "
        "at effort {effort} (your reasoning effort is unchanged)"
    ),
    "budget_exhausted_continue": (
        "the model spent its whole output budget thinking and produced nothing — re-asking "
        "it to continue the work"
    ),
    "budget_exhausted_no_output": (
        "the model spent its whole output budget thinking and produced no visible output "
        "({cause}) — retry, or switch to a model with a larger output budget"
    ),
    "output_limit_continuing": (
        "the model hit the output limit — continuing the answer where it stopped"
    ),
    "output_limit_cut_off": (
        "the model hit the output limit — this answer is cut off, and the rest was never "
        "sent — ask again to continue, or narrow the request"
    ),
    "output_limit_tool_call": (
        "the model hit the output limit mid tool call — nothing was executed; re-asking it "
        "to re-emit the call in smaller pieces"
    ),
    "output_limit_before_call": (
        "the model hit the output limit before the call ran — nothing was executed; "
        "re-asking it to re-issue the call as it is"
    ),
    "empty_reply": (
        "the model's reply arrived empty (no text, no tool calls) — re-asking it for its "
        "next action or finish"
    ),
    "repeated_tool_errors": "Repeated tool errors — requesting a different approach.",
    "kept_going_stopped": (
        "This turn was stopped because it kept being asked to keep going; the pending "
        "message was dropped. Reply with what to do next."
    ),
    # session/session.py (enforcement slice C)
    "image_unsupported": (
        "{model} does not accept images — the images in this conversation are omitted "
        "from its context. Switch to a model that accepts images to see them again."
    ),
    "request_too_large_dropped": (
        "The provider refused the request as too large. Some screenshots have been "
        "dropped from the context and the size limit tightened — send your message again."
    ),
    "too_large_run_compact": (
        "This conversation is too large for the provider to accept. Run /compact to "
        "summarise it, or start a new conversation."
    ),
    "too_large_images_removed": (
        "The conversation is too large for this provider even after dropping screenshots, "
        "so images have been removed from the context to keep the session working. If it "
        "still fails, run /compact or start a new conversation."
    ),
    "image_rejected_history": (
        "The provider rejected an image in this conversation's history. Images have been "
        "dropped from the context so the session keeps working — send your message again."
    ),
    "stopped_session_held": (
        "Work arriving into this stopped session is being held, not run. The held items "
        "are read on the next turn."
    ),
    "auto_continuation_limit": "Auto-continuation limit ({limit}) reached; stopping.",
    "steering_not_saved": (
        "steering command {command_id} was not saved: {reason}; retry with the same " "command ID"
    ),
    "unsaved_asides": "{count} aside(s) you added to the conversation",
    "unsaved_receipts": "{count} shell command record(s)",
    "unsaved_records": "{summary} could not be saved before the conversation closed",
    "model_info_incomplete": (
        "Saved model information is incomplete; using {model}. This selection is now "
        "saved for future resumes."
    ),
    "model_info_incomplete_headline": "Model selected",
    "model_default_changed": (
        "keeping {model}; default changed for new sessions. /model saved adopts it here"
    ),
    "model_default_changed_headline": "Model unchanged",
    "effort_default_changed": (
        "reasoning effort default changed for new sessions ({effort}); /model saved "
        "adopts it here"
    ),
    "effort_default_changed_headline": "Effort default changed",
}


def _catalogue() -> dict[str, str]:
    return catalogues.load_catalogue("en", NAMESPACE)


def test_catalogue_values_are_the_replaced_literals() -> None:
    catalogue = _catalogue()
    assert set(catalogue) == {f"{NAMESPACE}.{key}" for key in REPLACED_LITERALS}
    for key, literal in REPLACED_LITERALS.items():
        assert catalogue[f"{NAMESPACE}.{key}"].replace("''", "'") == literal, key


def test_every_key_carries_context() -> None:
    sidecar = catalogues.context("en", NAMESPACE)
    assert set(sidecar) == set(_catalogue())
    assert all(
        all(entry.get(field) for field in ("role", "tone", "note")) for entry in sidecar.values()
    )
    assert all("EN-only" in entry["note"] for entry in sidecar.values())


def test_notice_event_carries_code_and_params_beside_text() -> None:
    event = _notice(keys.budget_exhausted_retry(effort="low"))
    assert isinstance(event, NoticeEvent)
    assert event.code == "tui.notices.budget_exhausted_retry"
    assert event.params == {"effort": "low"}
    assert event.kind == "warning"
    assert event.text == (
        "model spent its whole output budget thinking and produced nothing — retrying once "
        "at effort low (your reasoning effort is unchanged)"
    )


_SAMPLE_PARAMS = {
    "budget_exhausted_retry": {"effort": "low"},
    "budget_exhausted_no_output": {"cause": "a stop was requested"},
    "image_unsupported": {"model": "gpt-x"},
    "auto_continuation_limit": {"limit": 8},
    "steering_not_saved": {"command_id": "c1", "reason": "busy"},
    "unsaved_asides": {"count": 2},
    "unsaved_receipts": {"count": 1},
    "unsaved_records": {"summary": "2 aside(s) you added to the conversation"},
    "model_info_incomplete": {"model": "gpt-x"},
    "model_default_changed": {"model": "gpt-x"},
    "effort_default_changed": {"effort": "high"},
}


def test_four_digit_counts_keep_plain_digits() -> None:
    """The legacy f-string rendered ``1000``; ICU's number path renders ``1,000``.

    Counts are therefore passed as strings at the call sites, and this pin makes
    a future int caller fail loudly instead of silently changing the output.
    """
    message = keys.unsaved_asides(count="1000")
    assert messages.render(message.code, message.params, locale="en") == (
        "1000 aside(s) you added to the conversation"
    )


def test_an_uncarried_locale_falls_back_to_english() -> None:
    for key in REPLACED_LITERALS:
        message = getattr(keys, key)(**_SAMPLE_PARAMS.get(key, {}))
        en = messages.render(message.code, message.params, locale="en")
        assert messages.render(message.code, message.params, locale="fr") == en
        assert _notice(message).text == en
