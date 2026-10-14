"""The C1 slice: `wire.errors.agents` on the wire.

What this file pins, and why each pin exists:

1. **`_publication_detail` keeps the code VERBATIM.** The publication body is
   the one place in this namespace that already carried a machine `code` (the
   key the renderer switches on — RFC §2.6's "only stable part"): the
   conversion moved the sentence to `text`/`params.error` and kept `details`
   side-by-side, and this asserts the renderer's contract survives shape
   changes. `message` must NOT come back.
2. **Byte-identity samples.** The ICU-trap renders (an apostrophe-quoted
   param — `''{variable_key}''` — and quote-bearing sentences) come back as
   the pre-change f-strings read, character for character.
3. **The pass-through key.** Lower-layer prose (`str(e)`, `ORG_LOGIN_REMEDY`,
   `org_destination_refused_sentence`) rides `agents_refusal` unaltered.
4. **Envelope params are JSON-safe.** The UUID class of bug — a
   non-serializable param inside the envelope body — is pinned where it bit:
   the schedule/agent UUID sites pass `str()`, and the envelope round-trips
   `json.dumps` with the rendered text intact.
"""

from __future__ import annotations

import json
import uuid

from local_operator.i18n import catalogues
from local_operator.i18n.keys import wire_errors_agents as w
from local_operator.i18n.text import tr
from local_operator.server.routes.agents import _publication_detail

_t = tr("wire.errors.agents")


def test_publication_detail_keeps_its_code_and_moves_the_sentence() -> None:
    detail = _publication_detail(
        "not_owner", "Someone else owns that listing.", {"field": "hub_id"}
    )
    assert detail["code"] == "not_owner"
    assert detail["params"] == {"error": "Someone else owns that listing."}
    assert detail["text"] == "Someone else owns that listing."
    assert detail["details"] == {"field": "hub_id"}
    assert "message" not in detail


def test_byte_identity_samples_are_char_for_char() -> None:
    samples = [
        (w.agent_not_found_id(agent_id="a1"), "Agent with ID a1 not found"),
        (w.agent_not_found(), "Agent not found"),
        (w.agent_not_found_wrapped(error="KeyError: 'x'"), "Agent not found: KeyError: 'x'"),
        (w.radient_api_key_required(), "RADIENT_API_KEY is required"),
        (w.error_uploading_agent(error="boom"), "Error uploading agent to Radient: boom"),
        (
            w.page_out_of_bounds(page="4", total_pages="1"),
            "Page 4 is out of bounds. Total pages: 1",
        ),
        # The ICU apostrophe trap: doubled quotes in the catalogue render as
        # the single quotes the f-string produced around the substituted param.
        (
            w.variable_not_found(variable_key="my_var", agent_id="a1"),
            "Execution variable 'my_var' not found for agent a1",
        ),
        (
            w.job_not_found_id(job_id="j1"),
            'Job with ID "j1" not found',
        ),
        (
            w.job_cannot_cancel(job_id="j1"),
            "Job j1 cannot be cancelled (already completed, failed, or previously cancelled)",
        ),
        (
            w.team_tenant_mismatch(owner="org-a", tenant_id="org-b"),
            "That team belongs to organization org-a, not org-b. Check the tenant.",
        ),
        (
            w.system_prompt_required(),
            "Request body must contain 'system_prompt' field",
        ),
        (w.publish_local_failure(), "This agent could not be published from this machine."),
        (
            w.listing_id_required(),
            "An update needs the id of the listing to update: "
            "the agent document is not valid: hub_agent_id must not be empty.",
        ),
        (w.internal_server_error(), "Internal Server Error"),
    ]
    sources = catalogues.load_catalogue("en", "wire.errors.agents")
    for message, expected in samples:
        rendered = _t(message)
        assert rendered == expected, message.code
        assert message.code in sources


def test_dynamic_prose_passes_through_refusal() -> None:
    rendered = _t(w.agents_refusal(error="organization operations need a signed-in account."))
    assert rendered == "organization operations need a signed-in account."


def test_uuid_bearing_params_round_trip_json() -> None:
    # The real path: the route str()'s the UUID before the message is built.
    envelope = w.schedule_not_found_id(schedule_id=str(uuid.uuid4())).envelope()
    payload = json.dumps(envelope)  # must not raise: the class of bug this pins
    assert uuid.UUID(json.loads(payload)["params"]["schedule_id"])
    assert json.loads(payload)["text"] == envelope["text"]
    assert envelope["text"].startswith("Schedule with ID ")
