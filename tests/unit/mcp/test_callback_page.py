"""The OAuth callback page: the contracts a redesign must not quietly break.

Deliberately not a snapshot suite. What is asserted here is the handful of
properties that are load-bearing rather than aesthetic — the voice boundary
around a third party's text, the identity mark never being conditional, the
zero-network rule, and the HTTP framing — because those are the ones whose
loss would be invisible in review. Spacing, radii and colour values are judged
by looking at rendered frames, not by a test.
"""

from __future__ import annotations

import html
import re

import pytest

from local_operator.mcp.callback_page import callback_response, render_callback_page


class TestZeroNetwork:
    """The page must render on a machine that cannot reach anything.

    It is served at the end of an OAuth flow, sometimes through a proxy that
    has not been authorized yet, and a permissions page that phones out is the
    wrong thing to hand someone. Every state is checked because the constraint
    is about the document, not about one branch of it.
    """

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"tone": "success", "server": "https://srv.example/mcp"},
            {"tone": "danger", "server": "https://srv.example/mcp", "provider_message": "denied"},
            {"tone": "neutral", "closable": False},
            # A caller string that CONTAINS the tokens the assertions look for.
            # It is escaped into a text node and cannot fetch, so the invariant
            # is about the document we author, not about what a provider says
            # inside it — and the assertions have to subtract it or they fail on
            # the very payload the escaping test proves is a real input.
            {
                "tone": "danger",
                "server": "https://srv.example/mcp",
                "provider_message": '<img src=x onerror="alert(1)"> see //cdn.example/x',
            },
        ],
    )
    def test_no_external_reference_of_any_kind(self, kwargs) -> None:
        page = render_callback_page("Title", "Detail.", **kwargs)
        # Everything a caller handed us, escaped exactly as the page escapes it,
        # removed before the document is examined.
        authored = page
        for supplied in (kwargs.get("server"), kwargs.get("provider_message")):
            if supplied:
                authored = authored.replace(html.escape(str(supplied)), "")
        # The load-bearing assertion, because it is an INVARIANT rather than a
        # blocklist: the document we author has no URL-bearing attribute at all.
        # A token list alone lets protocol-relative references through — `<use
        # href="//cdn/s.svg#m">` inside the inline mark, for one — and every
        # such reference has to spell `src=` or `href=` to fetch anything.
        assert re.search(r"(?:src|href)\s*=", authored) is None
        assert "//" not in authored
        # The token list stays as a second net over CSS, which fetches without
        # attributes. `url(` covers `url()` (CSS forbids whitespace between the
        # ident and the paren, so `url (…)` does not fetch); `image-set(` is the
        # other function that takes a bare URL string.
        for forbidden in ("<link", "<script", "<img", "@import", "url(", "image-set("):
            assert forbidden not in authored, forbidden


class TestVoiceBoundary:
    """A provider's own words never speak in Local Operator's voice.

    ``error_description`` is arbitrary text from a query string rendered inside
    a card carrying our mark. Escaping stops it being an injection; only a
    visible seam stops a hostile provider borrowing our voice for a paragraph
    of branded instruction.
    """

    def test_provider_message_gets_its_own_labelled_trough(self) -> None:
        page = render_callback_page(
            "Authorization failed",
            "The provider did not grant this authorization.",
            tone="danger",
            provider_message="access_denied",
        )
        assert 'class="label">Provider response<' in page
        assert 'class="trough">access_denied<' in page
        # And is NOT spliced into the sentence we speak.
        assert "authorization.access_denied" not in page
        assert "grant this authorization. access_denied" not in page

    def test_provider_message_is_escaped(self) -> None:
        page = render_callback_page(
            "Authorization failed",
            "Detail.",
            tone="danger",
            provider_message='<img src=x onerror="alert(1)">',
        )
        assert "<img" not in page
        assert "&lt;img" in page

    def test_a_huge_provider_message_is_truncated_visibly(self) -> None:
        """The voice boundary needs an extent boundary too.

        The listener's 16 KiB head budget is otherwise the only bound, and a
        description that spends it renders a card thousands of pixels tall —
        pushing OUR sentence, the one that says what to do next, far below the
        fold on a page with no navigation. Truncation is visible rather than
        clipped in CSS so the page never hides text without saying so.
        """
        from local_operator.mcp.callback_page import _MAX_PROVIDER_MESSAGE

        page = render_callback_page(
            "Authorization failed", "Detail.", tone="danger", provider_message="x" * 16_000
        )
        rendered = re.search(r'class="trough">(x+…?)<', page)
        assert rendered is not None
        assert len(rendered.group(1)) == _MAX_PROVIDER_MESSAGE
        assert rendered.group(1).endswith("…")

    def test_a_short_provider_message_is_untouched(self) -> None:
        page = render_callback_page(
            "Authorization failed", "Detail.", tone="danger", provider_message="access_denied"
        )
        assert 'class="trough">access_denied<' in page
        assert "…" not in page

    def test_absent_provider_message_renders_no_empty_trough(self) -> None:
        page = render_callback_page("Authorized", "Detail.", tone="success")
        assert "Provider response" not in page

    @pytest.mark.parametrize("blank", ["", "   ", "\t\n "])
    def test_a_whitespace_only_provider_message_renders_no_trough(self, blank) -> None:
        """`error_description=%20%20%20` is truthy and arrives from the wire.

        Gating on the raw argument and rendering the stripped text produces a
        labelled empty box, which is the same defect as an empty trough with a
        different cause.
        """
        page = render_callback_page(
            "Authorization failed", "Detail.", tone="danger", provider_message=blank
        )
        assert "Provider response" not in page
        assert 'class="trough"></p>' not in page


class TestIdentityIsNotConditional:
    """The mark is `--ink` in every state, as `logo-mark.tsx` has it.

    Wiring it to the status colour rendered the logo at 1.31:1 in the neutral
    state — a ghost on the one page whose job is to tell a user that the
    software they trust received their grant.
    """

    def test_mark_colour_never_depends_on_tone(self) -> None:
        rule = re.search(r"\.mark \{[^}]*\}", render_callback_page("T", "D"))
        assert rule is not None
        assert "var(--ink)" in rule.group(0)
        assert "--tone" not in rule.group(0)

    @pytest.mark.parametrize("tone", ["success", "danger", "neutral"])
    def test_every_state_carries_the_same_mark(self, tone) -> None:
        page = render_callback_page("T", "D", tone=tone)
        assert 'class="mark"' in page
        assert "color: var(--ink)" in page


class TestOutcomeSurvivesWithoutColour:
    """Colour is redundant: the heading and the tab title carry the outcome."""

    @pytest.mark.parametrize(
        "title", ["Authorized", "Authorization failed", "No authorization code", "Nothing here"]
    )
    def test_title_leads_the_document_title(self, title) -> None:
        page = render_callback_page(title, "Detail.")
        assert f"<title>{title} · Local Operator</title>" in page
        assert f"<h1>{title}</h1>" in page


class TestHttpFraming:
    """One request per connection, then the socket goes away."""

    def test_headers_are_complete_and_length_matches_the_body(self) -> None:
        raw = callback_response("Authorized", "Detail.", tone="success")
        head, _, body = raw.partition(b"\r\n\r\n")
        assert head.startswith(b"HTTP/1.1 200 OK\r\n")
        assert b"Connection: close" in head
        assert b"Cache-Control: no-store" in head
        match = re.search(rb"Content-Length: (\d+)", head)
        assert match is not None
        declared = int(match.group(1))
        assert declared == len(body)

    def test_status_is_caller_chosen(self) -> None:
        raw = callback_response("Nothing here", "D.", status="404 Not Found", closable=False)
        assert raw.startswith(b"HTTP/1.1 404 Not Found\r\n")

    def test_closable_controls_the_close_line(self) -> None:
        assert "close this tab" in render_callback_page("T", "D")
        assert "close this tab" not in render_callback_page("T", "D", closable=False)
