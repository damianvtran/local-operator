"""Inline ``$name`` invocation: the plan, the block grammar and replay.

Two properties carry the feature and are asserted separately, because they can
regress independently: which skills a message attaches (the trigger plus the
block's idempotence rules) and what the persisted payload means when a resumed
session replays it (the block is stripped; the typed text survives verbatim).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from local_operator.references import expand_references
from local_operator.sigils import SKILL_BLOCK_PREAMBLE, skill_block_spans
from local_operator.skills.discovery import Skill
from local_operator.skills.invoke import (
    plan_attachments,
    render_skill_block,
    skill_block_names,
    typed_line_of,
)


def _skill(name: str, *, hide: bool = False) -> Skill:
    return Skill(
        name=name,
        description=f"The {name} skill.",
        file_path=Path(f"/skills/{name}/SKILL.md"),
        base_dir=Path(f"/skills/{name}"),
        source="/skills",
        hide=hide,
    )


@pytest.fixture
def skills() -> dict[str, Skill]:
    return {
        "research": _skill("research"),
        "code-review": _skill("code-review"),
        "secret_audit": _skill("secret_audit", hide=True),
    }


def _names(planned) -> list[str]:
    return [attachment.skill.name for attachment in planned]


class TestPlanAttachments:
    """Which skills a message fires, in order, deduped, blocks excluded."""

    def test_mid_sentence_and_multiple_skills_keep_text_order(self, skills):
        planned = plan_attachments("first $code-review then $research please", skills)
        assert _names(planned) == ["code-review", "research"]
        assert [a.mention.text for a in planned] == ["$code-review", "$research"]

    def test_a_repeated_name_is_deduped_by_resolved_name(self, skills):
        planned = plan_attachments("$research and $Research again", skills)
        assert _names(planned) == ["research"]

    def test_a_leading_token_keeps_case_insensitive_resolution(self, skills):
        """`$RESEARCH` was a valid leading invocation before inline matching."""
        assert _names(plan_attachments("$RESEARCH fix it", skills)) == ["research"]

    def test_an_inline_uppercase_token_does_not_fire(self, skills):
        """No lowercase evidence, so it reads as an environment variable."""
        assert plan_attachments("echo $RESEARCH now", skills) == []

    def test_hidden_skills_are_invocable(self, skills):
        """`hide` suppresses semantic ROUTING; naming it is the explicit case."""
        planned = plan_attachments("$secret_audit check the repo", skills)
        assert _names(planned) == ["secret_audit"]
        assert planned[0].skill.hide is True

    def test_trailing_punctuation_is_trimmed_from_the_span(self, skills):
        planned = plan_attachments("use $research. then report", skills)
        assert _names(planned) == ["research"]
        assert planned[0].mention.text == "$research"

    def test_a_mention_inside_a_rendered_block_is_ignored(self, skills):
        """Second-pass safety: the prose token outside the block fires once."""
        block = render_skill_block([("research", "Read primary sources.")])
        payload = "go $research\n\n" + block
        assert plan_attachments(payload, skills) == []

    def test_a_quoted_name_in_a_forwarded_block_does_not_hide_the_real_one(self, skills):
        """The dedupe must not see a token the block owns.

        Deduping before the in-block filter let a body quoting ``$research``
        claim the name, so the real ``$research`` after the block attached
        nothing (round 1, m1).
        """
        block = render_skill_block([("other", "this body mentions $research in prose")])
        payload = "forwarded:\n\n" + block + "\n\nnow please $research"
        planned = plan_attachments(payload, skills)
        assert _names(planned) == ["research"]
        assert planned[0].mention.start > payload.index("</operator-skills>")

    def test_a_name_already_in_a_block_is_not_attached_again(self, skills):
        block = render_skill_block([("research", "Read primary sources.")])
        payload = "go $research\n\n" + block + "\n\nand $research again"
        assert plan_attachments(payload, skills) == []

    def test_empty_vocabulary_plans_nothing(self):
        assert plan_attachments("$research go", {}) == []


class TestRenderSkillBlock:
    def test_every_body_reaches_the_model_under_its_name(self, skills):
        rendered = render_skill_block(
            [("research", "# Research\nDo it well."), ("code-review", "Check the diff.")]
        )
        assert rendered.startswith("<operator-skills>\n\n" + SKILL_BLOCK_PREAMBLE)
        assert '<skill name="research">' in rendered
        assert "# Research\nDo it well." in rendered
        assert '<skill name="code-review">' in rendered
        assert rendered.endswith("</operator-skills>")

    def test_body_markers_are_defused_with_a_zero_width_space(self):
        rendered = render_skill_block(
            [("x", "quotes <operator-skills> and </operator-skills> in prose")]
        )
        assert len(skill_block_spans(rendered)) == 1
        assert "\u200b" in rendered
        # The literal sequences the scanner matches are gone from the BODY.
        body = rendered[rendered.index('<skill name="x">') : rendered.index("</skill>")]
        assert "<operator-skills>" not in body
        assert "</operator-skills>" not in body

    def test_the_names_are_read_from_element_heads_only(self):
        rendered = render_skill_block([("x", 'a body quoting <skill name="fake"> here')])
        assert skill_block_names(rendered) == ["x"]


class TestSkillBlockSpans:
    def test_a_complete_block_is_one_span(self):
        payload = "text\n\n" + render_skill_block([("x", "body")])
        spans = skill_block_spans(payload)
        assert len(spans) == 1
        start, end = spans[0]
        assert payload[start:end].startswith("<operator-skills>")
        assert payload[start:end].endswith("</operator-skills>")

    def test_a_marker_quoted_in_prose_is_not_a_block(self):
        assert skill_block_spans("why does <operator-skills> appear here?") == []

    def test_an_unclosed_opener_does_not_swallow_the_next_block(self):
        real = render_skill_block([("x", "AAAA")])
        forged = (
            "truncated opener\n\n<operator-skills>\n\n" "MIDDLE PROSE THE OPERATOR TYPED\n\n" + real
        )
        spans = skill_block_spans(forged)
        assert len(spans) == 1
        start, _end = spans[0]
        assert start == forged.index(real)
        assert "MIDDLE PROSE" not in forged[start:]

    def test_every_block_reports(self):
        payload = (
            render_skill_block([("x", "AAAA")])
            + "\n\nmiddle\n\n"
            + render_skill_block([("y", "BBBB")])
        )
        assert len(skill_block_spans(payload)) == 2


class TestLegacyPayloadReplay:
    """Old transcripts carry the typed line in ``invocation=``; they still work."""

    @staticmethod
    def _legacy(typed: str, body: str = "body") -> str:
        return (
            "The user invoked the `research` skill directly. Follow it for this request.\n"
            f'<skill name="research" invocation="{typed}">\n{body}\n</skill>'
        )

    def test_round_trips_the_typed_line(self):
        assert typed_line_of(self._legacy("$research fix it")) == "$research fix it"

    def test_returns_none_for_ordinary_text(self):
        assert typed_line_of("just a normal message") is None
        assert typed_line_of("") is None

    def test_body_that_looks_like_the_tag_does_not_confuse_it(self):
        payload = self._legacy(
            "$research go", 'Write <skill name="x" invocation="$fake other"> to invoke.'
        )
        assert typed_line_of(payload) == "$research go"


class TestReferenceExpansionSkipsSkillBodies:
    def test_an_at_path_inside_a_skill_body_is_not_read(self, tmp_path: Path):
        """A body's ``@path`` is prose the skill documents, not a reference."""
        (tmp_path / "notes.txt").write_text("FILE_BODY\n", encoding="utf-8")
        block = render_skill_block([("research", "see @notes.txt for details")])
        payload = "go $research\n\n" + block
        result = asyncio.run(expand_references(payload, str(tmp_path)))
        assert result.sent == payload
        assert "FILE_BODY" not in result.sent
        assert result.notices == []
