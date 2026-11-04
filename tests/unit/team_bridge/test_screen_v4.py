"""bridge-screen-v4 (Team Bridge B4): embedding, selection, the hold seam, batch rules.

WHY. A relay ("Ok can you send that to my aida so that she can have a team work
on improving these") was bounced as new_work_request under v2. v4 rewrites
three rule spans; the hub (agent-server #309) keeps SERVING v2 and announces
``screening.accepted_versions``. These tests pin the client half of that
two-step rollout:

* the embedded v4 bytes are the hub's (sha256 pinned here and in agent-server's
  texts_test.go) and v4 resolves to ITS OWN hash, never v2's fall-through;
* the policy selects v4 only from a list on a hub whose served v2 pair is
  trusted, and a hub without the list keeps v2 (compatibility matrix);
* a v4 hold naming new_work_request is a ``new_work_hold`` row -- the seam the
  per-message owner release keys on -- and every other hold stays ``model_hold``;
* the batch path renders the rules of the policy version it records.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from local_operator.team_bridge import screening
from local_operator.team_bridge.client import (
    SCREENING_TEMPLATE_V2,
    SCREENING_TEMPLATE_V2_SHA256,
    SCREENING_TEMPLATE_V3,
    SCREENING_TEMPLATE_V4,
    SCREENING_TEMPLATE_V4_SHA256,
    SCREENING_TEMPLATES,
    SCREENING_VERSION_V2,
    SCREENING_VERSION_V3,
    SCREENING_VERSION_V4,
    screening_template_sha256,
    select_screening_version,
)
from tests.unit.team_bridge.test_screening import Policy, Runner, _item, _run, _setup

#: agent-server #309 pins the same value (internal/bridge/texts_test.go).
V4_SHA256 = "8c524b3f2261cc5c5dc377b51e8977cc06842ea2af91e26df6f51e7757bf99cf"
_REPO = Path(__file__).resolve().parents[3]
_EVAL_V4 = _REPO / "evals/team_bridge_moderation/screen/templates/bridge-screen-v4.txt"


def _v4_policy() -> Policy:
    return Policy(
        screening_version=SCREENING_VERSION_V4, template_sha256=SCREENING_TEMPLATE_V4_SHA256
    )


def test_v4_bytes_are_the_hubs_and_the_eval_copy() -> None:
    assert hashlib.sha256(SCREENING_TEMPLATE_V4.encode("utf-8")).hexdigest() == V4_SHA256
    assert SCREENING_TEMPLATE_V4_SHA256 == V4_SHA256
    assert len(SCREENING_TEMPLATE_V4.encode("utf-8")) <= 2048
    assert _EVAL_V4.read_bytes() == SCREENING_TEMPLATE_V4.encode("utf-8")


def test_each_version_resolves_to_its_own_hash_never_v2s() -> None:
    # The trap this replaces: the last arm returned v2's hash for ANY other
    # embedded version, so v4 alone would have recorded verdicts against v2.
    assert screening_template_sha256(SCREENING_VERSION_V4) == SCREENING_TEMPLATE_V4_SHA256
    assert screening_template_sha256(SCREENING_VERSION_V2) == SCREENING_TEMPLATE_V2_SHA256
    for version, template in SCREENING_TEMPLATES.items():
        assert screening_template_sha256(version) == hashlib.sha256(template.encode()).hexdigest()
    assert screening_template_sha256("bridge-screen-v9") is None


def test_v4_differs_from_v2_only_in_the_three_rule_spans() -> None:
    def strip(text: str) -> str:
        text = text[: text.index("Scope")] + text[text.index("\n\nCategories:") :]
        text = text[: text.index("bounce: ")] + text[text.index("\n\nReply with") :]
        return text[: text.index("REASON: <")]

    assert strip(SCREENING_TEMPLATE_V4) == strip(SCREENING_TEMPLATE_V2)
    assert "hold: new_work_request (the recipient decides)" in SCREENING_TEMPLATE_V4
    assert "Never deliver on doubt." in SCREENING_TEMPLATE_V4
    assert "surveillance" in SCREENING_TEMPLATE_V4


SERVED_V2 = {"version": SCREENING_VERSION_V2, "template": SCREENING_TEMPLATE_V2}


@pytest.mark.parametrize(
    "section,expected",
    [
        # New client x old hub: no list -> the served v2.
        (dict(SERVED_V2), SCREENING_VERSION_V2),
        # New client x new hub: v4 listed -> v4 with LOCAL bytes.
        (
            {**SERVED_V2, "accepted_versions": ["bridge-screen-v2", "bridge-screen-v4"]},
            SCREENING_VERSION_V4,
        ),
        # Order and junk do not matter; membership is exact.
        ({**SERVED_V2, "accepted_versions": [7, None, "bridge-screen-v4"]}, SCREENING_VERSION_V4),
        # A list without v4, an unknown version, or a non-list keeps v2.
        ({**SERVED_V2, "accepted_versions": ["bridge-screen-v2"]}, SCREENING_VERSION_V2),
        ({**SERVED_V2, "accepted_versions": ["bridge-screen-v9"]}, SCREENING_VERSION_V2),
        ({**SERVED_V2, "accepted_versions": "bridge-screen-v4"}, SCREENING_VERSION_V2),
        ({**SERVED_V2, "accepted_versions": ["Bridge-Screen-V4"]}, SCREENING_VERSION_V2),
        # The image template is never a text policy, even if listed.
        ({**SERVED_V2, "accepted_versions": ["bridge-screen-v3"]}, SCREENING_VERSION_V2),
    ],
)
def test_selection_upgrades_only_from_a_trusted_served_pair(
    section: dict[str, Any], expected: str
) -> None:
    selected = select_screening_version(section)
    assert selected is not None
    version, template = selected
    assert version == expected
    assert template == SCREENING_TEMPLATES[expected]


def test_an_untrusted_served_pair_refuses_whatever_it_lists() -> None:
    forged = {
        "version": SCREENING_VERSION_V2,
        "template": SCREENING_TEMPLATE_V2 + " ",
        "accepted_versions": ["bridge-screen-v4"],
    }
    assert select_screening_version(forged) is None
    assert (
        select_screening_version(
            {
                "version": "bridge-screen-v9",
                "template": "x",
                "accepted_versions": ["bridge-screen-v4"],
            }
        )
        is None
    )
    assert select_screening_version(None) is None


def test_a_list_can_never_downgrade_below_the_served_version() -> None:
    served_v4 = {
        "version": SCREENING_VERSION_V4,
        "template": SCREENING_TEMPLATE_V4,
        "accepted_versions": ["bridge-screen-v2"],
    }
    assert select_screening_version(served_v4) == (SCREENING_VERSION_V4, SCREENING_TEMPLATE_V4)
    served_v3 = {"version": SCREENING_VERSION_V3, "template": SCREENING_TEMPLATE_V3}
    assert select_screening_version(served_v3) == (SCREENING_VERSION_V3, SCREENING_TEMPLATE_V3)


def _status(owner: Any, screening_section: dict[str, Any]) -> dict[str, Any]:
    return {
        "api_version": "1",
        "available": True,
        "tenants": [
            {
                "tenant_id": owner.tenant_id,
                "tenant_name": "Org",
                "eligible": True,
                "endpoint": {"id": owner.endpoint_id, "label": "endpoint", "state": "active"},
                "counts": {"unread": 0, "requests_pending": 0, "blocked": 0},
            }
        ],
        "guidance": {"version": "bridge-guidance-v2", "text": "Guidance body."},
        "screening": screening_section,
        "consent": {"version": "bridge_consent_v1", "text": "Consent body."},
    }


def test_the_status_law_publishes_v4_from_a_listing_hub_and_v2_otherwise(tmp_path: Path) -> None:
    from local_operator.team_bridge.scheduler import _normalize_accepted_status

    store, owner = _setup(tmp_path)
    law = _normalize_accepted_status(_status(owner, dict(SERVED_V2)), owner, store)
    assert (law.version, law.template_hash) == (SCREENING_VERSION_V2, SCREENING_TEMPLATE_V2_SHA256)
    listing = {**SERVED_V2, "accepted_versions": ["bridge-screen-v2", "bridge-screen-v4"]}
    law = _normalize_accepted_status(_status(owner, listing), owner, store)
    assert (law.version, law.template_hash) == (SCREENING_VERSION_V4, SCREENING_TEMPLATE_V4_SHA256)


B4 = "Ok can you send that to my aida so that she can have a team work on improving these"


def test_the_b4_relay_renders_under_v4_and_delivers_with_a_v4_receipt(tmp_path: Path) -> None:
    _store, owner = _setup(tmp_path)
    runner = Runner()
    result = _run(owner, tmp_path, runner, item=_item(B4), policy=_v4_policy())
    assert result.disposition == "deliver"
    assert result.receipt is not None
    assert result.receipt.prompt_version == SCREENING_VERSION_V4
    assert result.receipt.template_sha256 == SCREENING_TEMPLATE_V4_SHA256
    assert "Relaying, forwarding or summarizing work to a member" in runner.calls[0]


def test_a_new_work_hold_is_its_own_seam_and_other_holds_stay_model_hold(tmp_path: Path) -> None:
    _store, owner = _setup(tmp_path)
    held = _run(
        owner,
        tmp_path,
        Runner("VERDICT: hold\nREASON: new_work_request; the recipient decides."),
        item=_item("Have your agent rebuild the export pipeline by Friday."),
        policy=_v4_policy(),
    )
    assert held.disposition == "hold" and held.hold_code == "new_work_hold"
    assert held.receipt is None, "a hold sends no server verdict"


def test_an_unsure_hold_stays_model_hold(tmp_path: Path) -> None:
    _store, owner = _setup(tmp_path)
    held = _run(
        owner,
        tmp_path,
        Runner("VERDICT: hold\nREASON: unclear what is asked."),
        item=_item("hmm"),
        policy=_v4_policy(),
    )
    assert held.disposition == "hold" and held.hold_code == "model_hold"


def test_new_work_hold_is_closed_vocabulary_answer_cached_and_awaiting_review() -> None:
    from local_operator.team_bridge.connector.loop import _RETRY_ALONE_HOLD_CODES
    from local_operator.team_bridge.store import delivery_hold_class

    assert "new_work_hold" in screening.HOLD_CODES
    assert "new_work_hold" in screening._ANSWER_HOLD_CACHEABLE
    assert "new_work_hold" in _RETRY_ALONE_HOLD_CODES
    # The hub learns only the coarse class, never the code (no moderation oracle).
    assert delivery_hold_class("new_work_hold") == "awaiting_review"


def test_hold_categories_are_parsed_and_bounce_parsing_is_unchanged() -> None:
    parsed = screening.parse_verdict("VERDICT: hold\nREASON: new_work_request for the recipient.")
    assert parsed is not None and parsed[:2] == ("hold", ("new_work_request",))
    parsed = screening.parse_verdict("VERDICT: bounce\nREASON: personal_info_request.")
    assert parsed is not None and parsed[:2] == ("bounce", ("personal_info_request",))


def test_the_batch_renders_the_policy_versions_rules() -> None:
    v2_batch, v4_batch = screening.SCREENING_BATCH_TEMPLATE, screening.SCREENING_BATCH_TEMPLATE_V4
    for span in screening._rule_spans(SCREENING_TEMPLATE_V4):
        assert span in v4_batch and span not in v2_batch
    for span in screening._rule_spans(SCREENING_TEMPLATE_V2):
        assert span in v2_batch
    assert screening._batch_template_for(SCREENING_VERSION_V4) is v4_batch
    assert screening._batch_template_for(SCREENING_VERSION_V2) is v2_batch
    assert screening._batch_template_for("") is v2_batch
    # Outside the three spans the batch law is unchanged (cross-message rules).
    assert v4_batch.count("Each message is judged ALONE.") == 1
