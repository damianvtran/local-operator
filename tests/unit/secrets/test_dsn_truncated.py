"""A connection string cut before its host still carries a credential.

**The gap this file closes.** Every rule that can carry a DSN requires the
``@host`` tail: ``_DSN_PATTERN``/``_DSN_PATTERN_PLAIN`` match on the literal
``@`` plus a host lookahead, and the named-URL fallback decides through
``_url_contains_credential``, whose two userinfo patterns also start at ``://``
and END at ``@``. The at-sign is the match condition at both layers, so a value a
line-wrapped log, a truncated read or a ``head -c`` cut before its host is
proposed to no rule and is published in full.

**Why only the provable subset is decided.** A truncated value has no host, so
the rule has to read the userinfo alone. Three signals make the credential
provable: a KNOWN credential scheme, an explicit ``user:`` separator, and a
password run that is NOT all digits. The digit signal is the line between a
credential and a port — in a URL authority a port is always digits — so
``postgres://dbhost:5432/prod`` stays readable while ``postgres://appuser:S3cret``
is masked. ``http``/``https`` are excluded from the scheme list for the same
reason: ``http://localhost:8080`` is where a non-numeric reading would be wrong.

**What this file pins as NOT decided** (recorded residuals, not oversights): a
bare ``x:y`` with no scheme — ``dbhost:5432`` and ``appuser:S3cret`` are the same
spelling with no in-band signal — an all-digit password, and a truncation that
also cut the user. Each has a test asserting it stays readable, so a later
widening has a row to argue against.

Values are built from parts, following this directory's convention: the file is
read by agents through the very pass it describes.
"""

from __future__ import annotations

import json

import pytest

from local_operator import redaction_shapes as shapes
from local_operator.redaction_shapes import REDACTION_MARKER

#: A synthetic opaque password: not all digits, so it is a userinfo and not a port.
PASSWORD = "SyntheticPw7Qz2Lm"
USER = "appuser"


def _scrub(text: str) -> tuple[str, list[str]]:
    scrubbed, hits = shapes.scrub_secrets_with_hits(text)
    return scrubbed, sorted({hit.label for hit in hits})


def _password_survives(text: str) -> bool:
    scrubbed, _ = _scrub(text)
    return PASSWORD in scrubbed


class TestTruncatedConnectionStringIsMasked:
    @pytest.mark.parametrize(
        "text",
        [
            f"postgres://{USER}:{PASSWORD}",
            f"postgresql://{USER}:{PASSWORD}",
            f"mysql://{USER}:{PASSWORD}",
            f"mongodb://{USER}:{PASSWORD}",
            f"mongodb+srv://{USER}:{PASSWORD}",
            f"redis://:{PASSWORD}",
            f"DATABASE_URL=postgres://{USER}:{PASSWORD}",
            f"MONGO_DSN=mongodb://{USER}:{PASSWORD}",
        ],
        ids=[
            "postgres",
            "postgresql",
            "mysql",
            "mongodb",
            "mongodb-srv",
            "redis-empty-user",
            "env-postgres",
            "env-mongo",
        ],
    )
    def test_the_cut_password_is_masked(self, text: str) -> None:
        assert not _password_survives(text), "a truncated DSN password was published"

    def test_the_scheme_and_user_stay_readable(self) -> None:
        """An operator acts on which host and user; only the credential goes."""
        scrubbed, _ = _scrub(f"postgres://{USER}:{PASSWORD}")
        assert scrubbed == f"postgres://{USER}:{REDACTION_MARKER}"

    def test_the_hit_registers_the_password_not_the_user(self) -> None:
        _, hits = shapes.scrub_secrets_with_hits(f"postgres://{USER}:{PASSWORD}")
        values = {hit.value for hit in hits if hit.label == "dsn-password-truncated"}
        assert values == {PASSWORD}, values


class TestOrdinaryTextIsNotMasked:
    """The FP boundary: the rule decides from the userinfo, so these must survive."""

    @pytest.mark.parametrize(
        "text",
        [
            "postgres://dbhost:5432/prod",
            "http://localhost:8080/cb",
            "https://api.example.com",
            "SERVICE_URL=https://api.example.com",
            "CALLBACK_REDIRECT_URI=http://localhost:8080/cb",
            "max-width: 1200px",
            "time 12:30 and ratio 3:4",
            "connect to db.internal:5432 now",
        ],
        ids=[
            "postgres-host-port",
            "http-localhost-port",
            "https-endpoint",
            "service-url",
            "callback-uri",
            "css-width",
            "clock-and-ratio",
            "host-port-prose",
        ],
    )
    def test_no_credential_reading(self, text: str) -> None:
        scrubbed, _ = _scrub(text)
        assert scrubbed == text, scrubbed

    def test_an_ordinary_url_in_json_is_untouched(self) -> None:
        text = json.dumps({"SERVICE_URL": "https://api.example.com/health"})
        scrubbed, _ = _scrub(text)
        assert scrubbed == text, scrubbed


class TestWhitespaceBearingValuesAreNotClipped:
    """A token is not a word: the whole remainder must be captured, not clipped.

    The rule's run admits whitespace on purpose (review round 2, R2-1). When it
    stopped at the first space it published the TAIL of a multi-word passphrase
    while recording a ``dsn-password-truncated`` hit — a leak dressed as a
    containment — and CLIPPED a whitespace-bearing placeholder so the guard saw
    a broken prefix it could neither recognise nor back out, corrupting
    ``<your password here>`` into ``[redacted] password here>``.
    """

    @pytest.mark.parametrize(
        "reference",
        [
            "<" + "your password here" + ">",
            "[[" + "fixture name" + "]]",
            "${" + "CI TOKEN" + "}",
        ],
        ids=["angle-with-space", "bracket-with-space", "braced-with-space"],
    )
    def test_a_whitespace_placeholder_comes_back_intact(self, reference: str) -> None:
        """A whole reference is handed INTACT to the guard, which refuses it."""
        for text in (
            f"postgres://user:{reference}@db.invalid/prod",
            f"postgres://user:{reference}",
        ):
            scrubbed, _ = _scrub(text)
            assert scrubbed == text, scrubbed
            assert reference in scrubbed, "the placeholder must not be clipped"

    def test_a_multi_word_passphrase_masks_in_full(self) -> None:
        """No tail may survive: the whole run is the credential.

        Built from parts because this file is read through the very pass it
        describes -- a literal passphrase would be rewritten before ci could
        assert the tail was gone.
        """
        tail = "battery " + "staple"
        passphrase = "correct " + "horse " + tail
        text = "postgres://" + USER + ":" + passphrase
        scrubbed, labels = _scrub(text)
        assert labels == ["dsn-password-truncated"]
        assert passphrase not in scrubbed
        assert tail not in scrubbed, "the passphrase tail was published"
        assert scrubbed == "postgres://" + USER + ":" + REDACTION_MARKER

    def test_a_cut_at_the_separator_is_a_recorded_residual(self) -> None:
        """``…:pass@`` is owned by the rules above; unchanged here (base also open)."""
        text = f"postgres://{USER}:[redacted]@"
        scrubbed, _ = _scrub(text)
        assert scrubbed == text, scrubbed


class TestFullDsnAndNeighboursAreUnchanged:
    """The new rule must not disturb the rules that already worked."""

    def test_the_complete_dsn_still_masks_via_the_original_rule(self) -> None:
        text = f"postgres://{USER}:{PASSWORD}@db.internal.example.com/prod"
        scrubbed, labels = _scrub(text)
        assert labels == ["dsn-password"]
        assert PASSWORD not in scrubbed
        assert "db.internal.example.com" in scrubbed, "the host stays readable"

    def test_a_known_assignment_still_masks(self) -> None:
        scrubbed, labels = _scrub(f"password={PASSWORD}")
        assert labels == ["credential-assignment"]
        assert PASSWORD not in scrubbed

    def test_a_generic_assignment_still_masks(self) -> None:
        scrubbed, labels = _scrub(f"API_KEY={PASSWORD}")
        assert PASSWORD not in scrubbed
        assert labels, "some rule must claim it"


class TestRecordedResidualsStayUndecided:
    """What this rule deliberately does NOT claim — pinned so a widening argues here."""

    def test_an_all_digit_password_stays_unmasked(self) -> None:
        """A port is digits, so a digit tail is a host:port, not a credential."""
        text = f"postgres://{USER}:12345678"
        scrubbed, _ = _scrub(text)
        assert scrubbed == text, scrubbed

    def test_a_bare_user_colon_pass_with_no_scheme_stays_unmasked(self) -> None:
        """``dbhost:5432`` and ``appuser:S3cret`` are the same spelling."""
        for text in (f"{USER}:{PASSWORD}", "dbhost:5432"):
            scrubbed, _ = _scrub(text)
            assert scrubbed == text, scrubbed

    def test_a_truncation_that_cut_the_user_has_nothing_to_mask(self) -> None:
        for text in ("postgres://", "postgres://:5432"):
            scrubbed, _ = _scrub(text)
            assert scrubbed == text, scrubbed
