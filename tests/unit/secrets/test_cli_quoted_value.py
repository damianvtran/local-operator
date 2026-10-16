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
            "--secret \"<password>\"",
            "--password \"(redacted)\"",
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
