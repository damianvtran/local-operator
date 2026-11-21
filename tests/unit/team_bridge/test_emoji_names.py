"""Reaction emoji normalisation: shortcodes and literals -> one canonical key.

The hub (agent-server ``emoji.Normalize``) stores only fully-qualified RGI
sequences and refuses a shortcode as ``not_an_emoji``; these pin that the
runtime resolves agent-written shortcodes, qualifies a bare literal, and
refuses an unknown shortcode locally. The counts pin the generated table so a
regeneration must restate them.
"""

from __future__ import annotations

import hashlib
import json
from importlib import resources

import pytest

from local_operator.team_bridge import emoji_names
from local_operator.team_bridge.emoji_names import normalize_reaction_emoji


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("✅", "✅"),
        ("white_check_mark", "✅"),
        (":white_check_mark:", "✅"),
        ("  :White_Check_Mark:  ", "✅"),
        ("check_mark_button", "✅"),  # CLDR short name
        ("check-mark-button", "✅"),  # '-' folds to '_'
        ("check mark button", "✅"),  # ' ' folds to '_'
        ("+1", "👍"),
        (":+1:", "👍"),
        ("thumbsup", "👍"),
        ("thumbs_up", "👍"),
        ("-1", "👎"),
        ("eyes", "👀"),
        ("tada", "🎉"),
        ("rocket", "🚀"),
        ("heart", "\u2764\ufe0f"),
        ("red_heart", "\u2764\ufe0f"),
        ("\u2764", "\u2764\ufe0f"),  # literal missing its FE0F is qualified
        ("\u2764\ufe0f", "\u2764\ufe0f"),
        ("100", "💯"),
        ("hash", "#\ufe0f\u20e3"),
        ("👍🏽", "👍🏽"),  # skin-toned literal passes through
        # One fold with the hub's foldShortcode (agent-server #422 review R3).
        ("thumbs  up", "👍"),
        ("thumbs\tup", "👍"),
        (": rocket :", "🚀"),
        ("thumbs -_ up", "👍"),
        # The shared table: CLDR skin-tone names and both keycap names resolve
        # exactly as they do on the hub (#4239 review R2/R4).
        ("thumbs_up_medium_skin_tone", "👍🏽"),
        ("keycap_asterisk", "*\ufe0f\u20e3"),
        ("keycap_number_sign", "#\ufe0f\u20e3"),
        # A literal wrapped in shortcode colons is unwrapped (review R3).
        (":✅:", "✅"),
        (": ✅ :", "✅"),
    ],
)
def test_normalize_reaction_emoji_resolves_to_the_canonical_sequence(raw, expected):
    assert normalize_reaction_emoji(raw) == expected


@pytest.mark.parametrize(
    "raw",
    ["", "   ", "::", "not_a_real_emoji_name", ":nope:", "lol", "#", "keycap", "😀" * 33],
)
def test_normalize_reaction_emoji_refuses_unknown_or_unusable(raw):
    assert normalize_reaction_emoji(raw) is None


def test_every_table_value_is_fully_qualified_and_counts_are_pinned():
    doc = json.loads(
        resources.files("local_operator.team_bridge")
        .joinpath("emoji_names.json")
        .read_text(encoding="utf-8")
    )
    names, qualify = doc["names"], doc["qualify"]
    # Restate these when scripts/gen_emoji_names.py is re-run on new sources.
    assert len(names) == 4843
    assert len(qualify) == 1272
    assert doc["_sources"]["gemoji"]["tag"] == "v4.1.0"
    assert doc["_sources"]["emoji_test"]["version"] == "18.0"
    # No value is itself an unqualified form, so the two callers can never
    # send a key that a picker-emitted emoji would not match.
    assert not set(names.values()) & set(qualify)
    # Keys are already in the folded grammar the lookup uses.
    assert all(emoji_names.shortcode_key(k) == k for k in names)


def test_accepted_forms_hint_names_the_forms_and_examples():
    hint = emoji_names.ACCEPTED_FORMS_HINT
    for needle in ("✅", ":white_check_mark:", "+1", "eyes", "tada", "rocket", "heart"):
        assert needle in hint


#: sha256 of the name table's canonical serialization: every
#: ``"<name>\t<emoji>\n"`` line (emoji as UTF-8), sorted by name. agent-server
#: pins THE SAME VALUE as ``sharedTableSHA256`` in
#: ``internal/bridge/emoji/emoji_test.go`` for its ``aliases.txt``, so the
#: client refuses locally exactly what the hub would refuse. The two
#: generators and both pins must change together, in paired PRs.
SHARED_TABLE_SHA256 = "9ed0f5c2eb05ebecb0eabc40aab0631678e373f03e931d6bb035df648f100b66"


def test_name_table_matches_the_hub_digest():
    canonical = "".join(f"{k}\t{v}\n" for k, v in sorted(emoji_names.emoji_names().items()))
    assert hashlib.sha256(canonical.encode("utf-8")).hexdigest() == SHARED_TABLE_SHA256
