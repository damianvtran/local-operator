r"""A quote-delimited value after a credential flag must be judged, not skipped.

**The gap this file closes.** The ``cli-credential-flag`` shape exists for the
way a CLI takes a credential as a flag — ``--password``, ``--token``,
``--api-key``, ``--secret`` — with the flag name as context and the value after
it as the credential. Its value carrier was ``[^\s"']{3,}``, which REFUSES the
quote bytes. For the space spelling that is fatal rather than merely narrow: the
engine can only try to start the value at the OPENING QUOTE, that class refuses
that position, and the value alternative never fires at all. So
``deploy --token "…" --env prod`` was proposed to the guard by no rule and was
published in full — in the tool result, the durable transcript, the provider
history and the fork.

**Why the quotes are a carrier and not decoration.** They are how the value is
actually spelled wherever a shell or a JSON-logged command line records it. The
widening admits a quoted run, and the quote characters stay OUTSIDE the value
group so the mask rewrites the value and leaves the quotes: the same rule the
guarded renderer already follows for a JSON value's closing quote.

**The FP boundary is the guard, and it is unchanged.** The guard is handed the
INNER value, so every judgement it already made keeps deciding — a usage
placeholder carrying brackets (``--secret "NAME[=VALUE]"``), an expression
(``--token "$TOKEN"``), and a reference to a stored entry
(``--password "${CI_TOKEN}"``) all stay readable. The only value newly masked is
one the SAME judgement already masks when it is written unquoted, so this file
carries that unquoted control beside every quoted positive.

Values are built from parts, following the convention this directory already
uses: this file is read by agents through the very pass it describes, so a
credential-shaped literal in its own source would be a leak in THEIR transcript.
"""

from __future__ import annotations

import json

import pytest

from local_operator import redaction_shapes as shapes
from local_operator.redaction_shapes import REDACTION_MARKER

#: An opaque CLI credential: no vendor prefix and no long hex run, so the CLI
#: flag rule is the ONLY rule that can recognise it. A value some other shape
#: also catches would make a positive pass for the wrong reason.
OPAQUE = "Zx9Kp2-Qm7Lv4Rt8"

#: The same flag written the two ways a command line records it.
FLAG = "--token"


def _masked(text: str) -> tuple[str, list[str]]:
    """``scrub_secrets_with_hits`` is the composed pass: labels AND the text."""
    scrubbed, hits = shapes.scrub_secrets_with_hits(text)
    return scrubbed, sorted({hit.label for hit in hits})


def _value_published(text: str, value: str = OPAQUE) -> bool:
    """Whether ``value`` survives the pass — read from the ACTUAL output text.

    Asserting on the returned string rather than on a boolean derived from it is
    deliberate: a rule that fired but masked the wrong span would still satisfy a
    ``marker in out`` check, and this is exactly the file where the span matters.
    """
    scrubbed, _ = _masked(text)
    return value in scrubbed


class TestQuotedCliValueIsMasked:
    @pytest.mark.parametrize(
        "text",
        [
            FLAG + " " + OPAQUE,
            FLAG + " '" + OPAQUE + "'",
            FLAG + ' "' + OPAQUE + '"',
            "deploy " + FLAG + ' "' + OPAQUE + '" --env prod',
        ],
        ids=["bare-space", "single-quoted", "double-quoted", "quoted-mid-command"],
    )
    def test_the_value_is_gone_in_every_spelling(self, text: str) -> None:
        assert not _value_published(text), "the quoted CLI credential was published"

    def test_the_quotes_survive_the_mask(self) -> None:
        """The quotes are load-bearing context, not part of the credential.

        A carrier that captured the quote bytes into the value group would
        replace the quotes with the marker and unquote the surrounding text.
        """
        for quote in ("'", '"'):
            text = FLAG + " " + quote + OPAQUE + quote
            scrubbed, labels = _masked(text)
            assert "cli-credential-flag" in labels
            assert scrubbed == FLAG + " " + quote + REDACTION_MARKER + quote, scrubbed

    def test_a_quoted_value_containing_spaces_is_also_judged(self) -> None:
        """Quoting is what ALLOWS the space — the interior is still the value."""
        secret = OPAQUE + " and more"
        scrubbed, labels = _masked(FLAG + ' "' + secret + '"')
        assert "cli-credential-flag" in labels
        assert secret not in scrubbed

    def test_the_hit_registers_the_inner_value_not_the_quotes(self) -> None:
        """Containment depends on what gets registered, so pin the group."""
        _, hits = shapes.scrub_secrets_with_hits(FLAG + ' "' + OPAQUE + '"')
        values = {hit.value for hit in hits if hit.label == "cli-credential-flag"}
        assert values == {OPAQUE}, values


class TestQuotedReferencesAndPlaceholdersStayReadable:
    """The FP boundary: a quoted value the guard already refuses stays refused."""

    @pytest.mark.parametrize(
        "text",
        [
            FLAG + ' "NAME[=VALUE]"',
            FLAG + " 'NAME[=VALUE]'",
            '--secret "<password>"',
            '--password "(redacted)"',
        ],
        ids=["brackets-double", "brackets-single", "angle-brackets", "parens"],
    )
    def test_a_quoted_usage_placeholder_is_not_masked(self, text: str) -> None:
        scrubbed, _ = _masked(text)
        assert scrubbed == text, scrubbed

    @pytest.mark.parametrize(
        "text",
        [
            '--token "$TOKEN"',
            "--password '${CI_TOKEN}'",
            '--secret "${{ secrets.PROD_TOKEN }}"',
        ],
        ids=["dollar-token", "braced-ci-token", "github-expression"],
    )
    def test_a_quoted_expression_is_not_masked(self, text: str) -> None:
        scrubbed, _ = _masked(text)
        assert scrubbed == text, scrubbed

    @pytest.mark.parametrize(
        "text",
        [
            '--secret "GITHUB_TOKEN"',
            "--token 'OS_PROD2_ADMIN_PASSWORD'",
        ],
        ids=["store-name", "store-name-single"],
    )
    def test_a_quoted_reference_to_a_stored_entry_is_not_masked(self, text: str) -> None:
        scrubbed, _ = _masked(text)
        assert scrubbed == text, scrubbed


class TestUnquotedBehaviourIsUnchanged:
    """The widening must not move any spelling that already worked."""

    def test_the_bare_space_spelling_still_masks(self) -> None:
        assert not _value_published(FLAG + " " + OPAQUE)

    def test_the_bare_space_spelling_yields_the_same_text_as_the_quoted_one(self) -> None:
        bare, bare_labels = _masked(FLAG + " " + OPAQUE)
        quoted, quoted_labels = _masked(FLAG + ' "' + OPAQUE + '"')
        assert bare == FLAG + " " + REDACTION_MARKER
        assert bare_labels == quoted_labels == ["cli-credential-flag"]

    def test_a_non_latin_quoted_value_is_still_judged(self) -> None:
        secret = "\u4f60\u597d\u4e16\u754c\u30c6\u30b9\u30c8\u5b9f\u884c"
        scrubbed, labels = _masked(FLAG + ' "' + secret + '"')
        assert "cli-credential-flag" in labels
        assert secret not in scrubbed

    def test_an_unquoted_reference_is_still_readable(self) -> None:
        scrubbed, _ = _masked("--secret GITHUB_TOKEN")
        assert scrubbed == "--secret GITHUB_TOKEN"


class TestKnownValuePriorityIsIntact:
    """A known VALUE is masked by exact value first, quotes or not."""

    def test_a_known_value_inside_quotes_is_masked(self) -> None:
        known = "SyntheticKnownValue7Qz"
        scrubbed = shapes.scrub_values(FLAG + ' "' + known + '"', [known])
        assert known not in scrubbed
        assert REDACTION_MARKER in scrubbed

    def test_a_known_value_beside_a_quoted_flag_value_still_masks_both(self) -> None:
        known = "SyntheticKnownValue7Qz"
        text = FLAG + ' "' + known + '"'
        scrubbed, _ = shapes.scrub_secrets_with_hits(text, [known])
        assert known not in scrubbed


class TestNoNewFalsePositivesOnOrdinaryQuotedText:
    """Quoting is everywhere in JSON, so the widening's blast radius is pinned."""

    @pytest.mark.parametrize(
        "text",
        [
            json.dumps({"message": "token budget exceeded for this request"}),
            json.dumps({"note": "the password prompt appears once"}),
            'kubectl describe secret "app-config"',
            'git commit -m "add token refresh handler"',
            json.dumps({"token": "NAME"}),
        ],
        ids=["token-prose", "password-prose", "kubectl-name", "commit-message", "name-value"],
    )
    def test_ordinary_quoted_text_is_not_rewritten(self, text: str) -> None:
        scrubbed, _ = _masked(text)
        assert scrubbed == text, scrubbed


#: An opaque CLI credential built FROM PARTS, for the same reason the module's own
#: literals are: this file is read by agents through the pass it describes, so a
#: credential-shaped literal in its source would be a leak in THEIR transcript.
#: The two halves are joined at runtime so no single literal in the tree is the
#: value the tests assert on.
RESIDUAL_VALUE = "Zx9Kp2" + "-" + "Qm7Lv4Rt8"

#: A double-quoted value carrying the OTHER quote inside it, and the mirror image.
#: A shell records exactly this spelling whenever the credential itself contains a
#: quote character, and a JSON-logged command line is the other place it appears.
DQ_WITH_INNER_SQ = '"' + RESIDUAL_VALUE + "'" + RESIDUAL_VALUE + '"'
SQ_WITH_INNER_DQ = "'" + RESIDUAL_VALUE + '"' + RESIDUAL_VALUE + "'"


class TestAQuotedValueCarryingTheOtherQuoteIsMasked:
    """R1: the other quote byte must not break the value run.

    The value carrier excluded BOTH quote characters, so a value whose own bytes
    carry the opposite quote ended the run at that byte, the quoted alternative
    could never complete, no rule matched, and the value was published in full
    with no hit and no notice. Both directions are pinned because the carrier
    refuses the quote pair symmetrically.
    """

    @pytest.mark.parametrize(
        "text",
        [
            FLAG + " " + DQ_WITH_INNER_SQ,
            FLAG + " " + SQ_WITH_INNER_DQ,
        ],
        ids=["double-with-inner-single", "single-with-inner-double"],
    )
    def test_the_value_is_gone(self, text: str) -> None:
        scrubbed, labels = _masked(text)
        assert "cli-credential-flag" in labels, labels
        assert RESIDUAL_VALUE not in scrubbed, scrubbed

    def test_the_outer_quotes_survive_and_only_the_value_goes(self) -> None:
        """The mask keeps the delimiter and rewrites the whole interior."""
        scrubbed, _ = _masked(FLAG + " " + DQ_WITH_INNER_SQ)
        assert scrubbed == FLAG + ' "' + REDACTION_MARKER + '"', scrubbed

    def test_the_registered_value_is_the_whole_interior(self) -> None:
        _, hits = shapes.scrub_secrets_with_hits(FLAG + " " + DQ_WITH_INNER_SQ)
        values = {hit.value for hit in hits if hit.label == "cli-credential-flag"}
        assert values == {DQ_WITH_INNER_SQ.strip('"')}, values


class TestAnUnterminatedQuotedValueIsMasked:
    """R2: a quote with no closer must still mask the run.

    ``(?(q)(?P=q))`` required the closing quote to match, so an unterminated run
    failed the conditional, matched nothing and was published in full. A truncated
    transcript — an echoed line, a cut-off log tail — is exactly this shape.
    """

    @pytest.mark.parametrize(
        "text",
        [
            FLAG + ' "' + RESIDUAL_VALUE,
            FLAG + " '" + RESIDUAL_VALUE,
        ],
        ids=["unterminated-double", "unterminated-single"],
    )
    def test_the_value_is_gone(self, text: str) -> None:
        scrubbed, labels = _masked(text)
        assert "cli-credential-flag" in labels, labels
        assert RESIDUAL_VALUE not in scrubbed, scrubbed

    def test_the_opening_quote_survives(self) -> None:
        scrubbed, _ = _masked(FLAG + ' "' + RESIDUAL_VALUE)
        assert scrubbed == FLAG + ' "' + REDACTION_MARKER, scrubbed


class TestTheQuotedWideningKeepsItsFloorsAndControls:
    """The control surface R1/R2 must not move."""

    @pytest.mark.parametrize(
        "text,expected",
        [
            (FLAG + " " + RESIDUAL_VALUE, FLAG + " " + REDACTION_MARKER),
            (FLAG + ' "' + RESIDUAL_VALUE + '"', FLAG + ' "' + REDACTION_MARKER + '"'),
            (FLAG + " '" + RESIDUAL_VALUE + "'", FLAG + " '" + REDACTION_MARKER + "'"),
        ],
        ids=["bare-space-control", "double-quoted-control", "single-quoted-control"],
    )
    def test_the_q1_spellings_are_unchanged(self, text: str, expected: str) -> None:
        scrubbed, labels = _masked(text)
        assert scrubbed == expected, scrubbed
        assert labels == ["cli-credential-flag"], labels

    def test_a_sub_floor_quoted_value_is_untouched(self) -> None:
        """No floor was widened: a 2-char quoted value carries no ≥3-char secret."""
        for text in (FLAG + ' "Zx"', FLAG + " 'Zx'", FLAG + ' ""'):
            scrubbed, _ = _masked(text)
            assert scrubbed == text, scrubbed

    def test_a_duration_flag_is_not_a_credential(self) -> None:
        scrubbed, labels = _masked("--token-ttl=3600")
        assert scrubbed == "--token-ttl=3600", scrubbed
        assert labels == [], labels

    @pytest.mark.parametrize(
        "text",
        [
            FLAG + ' "NAME[=VALUE]"',
            FLAG + " 'NAME[=VALUE]'",
            '--token "$TOKEN"',
            "--password '${CI_TOKEN}'",
            '--secret "GITHUB_TOKEN"',
            "--token 'OS_PROD2_ADMIN_PASSWORD'",
        ],
        ids=[
            "placeholder-double",
            "placeholder-single",
            "expression",
            "brace-reference",
            "store-name",
            "store-name-single",
        ],
    )
    def test_the_guards_verdicts_still_decide_the_inner_value(self, text: str) -> None:
        scrubbed, _ = _masked(text)
        assert scrubbed == text, scrubbed

    def test_a_quoted_value_containing_a_space_is_judged(self) -> None:
        """The interior is the value, spaced spelling included."""
        secret = RESIDUAL_VALUE + " and more"
        scrubbed, labels = _masked(FLAG + ' "' + secret + '"')
        assert "cli-credential-flag" in labels
        assert secret not in scrubbed


class TestAnOpenQuoteWithNoCloserDoesNotEatProse:
    """QA round 1, Q1 (MAJOR): the unterminated branch must not run to EOL.

    An earlier cut closed R2 by letting the quoted run extend to a matching
    closer OR to end of line when there is none. That masked ordinary text
    whenever a credential flag was followed by an opening quote and no value had
    been opened at all — help/usage lines, docs, error text, a chained shell
    line — which is the "a mask the operator learns to distrust" failure the
    module docstring warns about. The unterminated branch is now bounded to a
    SINGLE non-whitespace token, so an opening quote ahead of prose matches
    nothing.

    These are the exact shapes QA measured, kept as regressions. The prose is
    literal text (it carries no credential), so it needs no parts-built literal.
    """

    @pytest.mark.parametrize(
        "text",
        [
            "usage: tool " + FLAG + ' "  the api token to send',
            "usage: tool " + FLAG + " '  the api token to use",
            'see --password "  the password to use',
            'error: --secret " requires a value',
            FLAG + ' " --env prod --verbose',
        ],
        ids=["usage-dq", "usage-sq", "password-prose", "error-line", "chained-flags"],
    )
    def test_prose_after_an_opening_quote_is_not_rewritten(self, text: str) -> None:
        scrubbed, labels = _masked(text)
        assert scrubbed == text, scrubbed
        assert labels == [], labels

    def test_the_same_line_with_a_real_single_token_value_still_masks(self) -> None:
        """The bound is the fix, not a retreat: a token is still a value."""
        scrubbed, labels = _masked(FLAG + ' "' + RESIDUAL_VALUE)
        assert "cli-credential-flag" in labels
        assert RESIDUAL_VALUE not in scrubbed


class TestAMultiWordUnterminatedValueIsADocumentedResidual:
    """No closer means no knowable end, so it is left readable rather than guessed.

    The rule cannot know where a multi-word unterminated value stops without a
    delimiter, and every spelling that guessed (end-of-line) over-masked prose
    (see the class above). The residual is stated and pinned here rather than
    chased, so a later round that widens the unterminated branch back to EOL has
    a row to argue against — it must not do so by resurrecting the over-mask.
    """

    @pytest.mark.parametrize(
        "text,expected",
        [
            (
                FLAG + ' "' + RESIDUAL_VALUE + " and more",
                FLAG + ' "' + REDACTION_MARKER + " and more",
            ),
            (
                FLAG + " '" + RESIDUAL_VALUE + " and more",
                FLAG + " '" + REDACTION_MARKER + " and more",
            ),
        ],
        ids=["multiword-unterminated-dq", "multiword-unterminated-sq"],
    )
    def test_the_leading_token_is_masked_and_the_tail_is_the_residual(
        self, text: str, expected: str
    ) -> None:
        """The FIRST token is a value and masks; the words after it are the residual.

        ``--token "<credential> and more`` with no closer bound the run to the one
        non-whitespace token, so the credential itself is gone. What stays readable
        is the tail after it — the honest residual, not a leak: a rule that does not
        know where the value ends cannot mask the words without a delimiter, and
        end-of-line was the over-mask this whole class exists to refuse.
        """
        scrubbed, labels = _masked(text)
        assert "cli-credential-flag" in labels, labels
        assert RESIDUAL_VALUE not in scrubbed, scrubbed
        assert scrubbed == expected, scrubbed
