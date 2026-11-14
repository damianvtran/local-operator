"""The E2 slice: `wire.errors.desktop` on the wire.

What this file pins, and why each pin exists:

1. **The envelope override.** Machine codes the released desktop app switches
   on by equality (`session_is_remote`, `runtime_busy`, `project_not_found`)
   ride VERBATIM as `code`, while the sentence comes from the catalogue — the
   reason `Msg.envelope(code=...)` exists at all. Renaming them to message ids
   would break every released client build.
2. **Byte-identity samples.** A handful of the migrated sentences, spelled the
   way the f-strings spelled them (including the em-dash spacing and the
   quoted `'{effort}'`), so a reworded catalogue value fails here before a
   reviewer has to diff two renderings by eye. The `effort_not_in_levels`
   sample is ALSO the ICU apostrophe-trap regression: `'{effort}'` reads as a
   literal in MessageFormat, so the catalogue carries doubled apostrophes.
3. **The runtime-unreachable drift guard.** `RUNTIME_UNREACHABLE_MESSAGE` is
   kept as a module constant for existing importers, and it must equal the
   catalogue render — one sentence, two spellings, pinned equal.
4. **The pass-through key.** `desktop_refusal` carries a lower layer's prose
   verbatim (params + text both equal to the input); every `str(error)`-shaped
   site in the slice leaves the wire text exactly as it was.
"""

from __future__ import annotations

from local_operator.i18n.keys import wire_errors_desktop as w


def test_machine_codes_ride_verbatim_through_the_override() -> None:
    envelope = w.requested_not_found().envelope(code="session_is_remote")
    assert envelope["code"] == "session_is_remote"
    assert envelope["params"] == {}
    assert envelope["text"] == "Requested session, profile, team or subscription not found"


def test_override_leaves_the_default_code_when_absent() -> None:
    assert w.requested_not_found().envelope()["code"] == ("wire.errors.desktop.requested_not_found")


def test_byte_identity_samples() -> None:
    # The em-dash and its single spaces ARE the f-string's output; the quotes
    # around the effort value are doubled apostrophes in the catalogue (ICU
    # quoting), which the runtime renders as plain quotes.
    assert (
        w.effort_not_in_levels(model="m", levels="low, high", effort="max").envelope()["text"]
        == "m accepts low, high \u2014 not 'max'"
    )
    assert (
        w.session_is_remote_write(noun="session", session_id="ab12", label="laptop").envelope()[
            "text"
        ]
        == "The session for ab12 lives on laptop, and this desktop cannot write it "
        "there yet; nothing was changed here."
    )
    assert (
        w.scope_name_too_long(max_length=str(64)).envelope()["text"]
        == "scope_name must be at most 64 characters."
    )
    # A param value that is itself repr()'d at the call site (the projects 404
    # quotes the key the way the old f-string did): quoting stays the CALLER's
    # choice, the catalogue carries the bare placeholder.
    assert (
        w.project_not_found(key=repr("x"), names="").envelope()["text"]
        == "no project with id or name 'x'"
    )


def test_runtime_unreachable_constant_matches_the_catalogue() -> None:
    from local_operator.server.routes.desktop_sessions import (
        RUNTIME_UNREACHABLE_MESSAGE,
    )

    assert (
        RUNTIME_UNREACHABLE_MESSAGE == w.runtime_unreachable().envelope()["text"]
    ), "the constant and the catalogue sentence have drifted"


def test_desktop_refusal_carries_lower_layer_prose_verbatim() -> None:
    envelope = w.desktop_refusal(error="filesystem said no").envelope(code="move_failed")
    assert envelope["code"] == "move_failed"
    assert envelope["params"] == {"error": "filesystem said no"}
    assert envelope["text"] == "filesystem said no"


def test_settings_intent_codes_ride_verbatim() -> None:
    """The settings-intent machine codes the renderer switches on.

    ``model_unknown``, ``move_exclusive`` and ``retry_budget_exhausted`` reuse the
    pass-through sentence (the runtime's own refusal) so no new copy is added for
    them; ``effort_unsupported`` reuses the ladder sentences; ``generation_moved``
    is the one new catalogued sentence (DELETE's 409).
    """
    for code in ("model_unknown", "move_exclusive", "retry_budget_exhausted"):
        envelope = w.desktop_refusal(error="the owner's words").envelope(code=code)
        assert envelope["code"] == code
        assert envelope["text"] == "the owner's words"
    assert w.runtime_unreachable().envelope(code="retry_budget_exhausted")["code"] == (
        "retry_budget_exhausted"
    )
    assert (
        w.effort_levels_missing(model="m").envelope(code="effort_unsupported")["code"]
        == "effort_unsupported"
    )
    moved = w.settings_intent_generation_moved().envelope(code="generation_moved", locale="en")
    assert moved["code"] == "generation_moved"
    assert moved["params"] == {}
    assert moved["text"] == "A newer change has replaced this one, so there is nothing to cancel."
