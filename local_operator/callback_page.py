"""The page the browser lands on at the end of an OAuth grant.

Served by both loopback listeners local-operator runs — the MCP authorization
flow (``local_operator.mcp.auth``) and the provider login flows
(``local_operator.providers.oauth.callback_server``). It is the only web
surface local-operator serves, and it is served at the worst possible moment:
the user has just handed a third party permission on their behalf and is
looking for confirmation that the thing they authorized
actually received it. A bare ``<h2>`` on a white page reads like a server
error even when it says "Authorized", so this page is drawn in the product's
own design language — the Local Operator marketing site's system
(``local-operator-site/docs/design-language.md``): warm paper, one accent,
hairlines instead of shadows, an old-style serif display over a humanist sans,
and a dark ramp for a user whose OS is dark.

Three constraints shape the implementation:

- **Zero network.** No font CDN, no stylesheet, no image request. The tab must
  render identically on a laptop that is offline or behind a proxy that has not
  been authorized yet, and a page that phones out at the end of an OAuth flow
  is the wrong thing to hand someone thinking about permissions. So: one
  document, inline ``<style>``, inline SVG, and a system-font stack in place of
  the three webfonts the site self-hosts. The serif stack is chosen so the
  first hit on every platform is an OLD-STYLE face in Fraunces' family of
  shapes (New York on macOS, Georgia on Windows, Palladio on Linux); Times is
  deliberately not in it, because Times is the one common serif that reads as
  *browser default* — the register this page exists to avoid.
- **One file, no build step.** The site is React + Tailwind v4; none of that can
  run here. The tokens below are copied from that system's ``@theme`` block
  with their names kept, so a change there is greppable here.
- **Both ramps, always.** ``prefers-color-scheme`` swaps the ramp via CSS
  variables exactly as the site's ``.dark`` block does. Every pair below is
  from the system's contrast tables: body ink is AAA on its ground in both
  ramps, and the dimmest text used (``ink-dim`` on ``surface``) is 5.43:1
  light, 5.11:1 dark.

The tone argument is not decoration. Success and failure of an authorization
are the one thing this page exists to distinguish, and the system's rule is
that colour is only ever spent on real semantics — so the 2px rule along the
card's top edge is the page's ONE spend, ``--success`` when a code arrived,
``--danger`` when the provider refused, and a plain hairline for the 404 a
browser's speculative fetch earns. The identity mark is never tinted: it is
``--ink`` in every state, as ``logo-mark.tsx`` has it, because the one thing
that must not look conditional on this page is the software's own identity.
"""

from __future__ import annotations

import html
from typing import Literal

Tone = Literal["success", "danger", "neutral"]

#: The Local Operator mark, traced in `local-operator-site`'s `logo-mark.tsx`.
#: Inlined rather than fetched (see the module docstring) and stroked in
#: ``currentColor``, which ``.mark`` pins to ``--ink`` in all three states.
_LOGO_MARK = (
    '<svg class="mark" viewBox="285 253 520 520" fill="none" stroke="currentColor" '
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false">'
    '<circle cx="490" cy="355" r="78" stroke-width="35"/>'
    '<circle cx="720" cy="326" r="42" stroke-width="29"/>'
    '<circle cx="368" cy="686" r="41" stroke-width="29"/>'
    '<path d="M370 645 V545 a90 90 0 0 1 90 -90 h60 a78 78 0 0 1 66 44" stroke-width="33"/>'
    '<path d="M590 498 V750" stroke-width="33"/>'
    '<path d="M600 492 C 646 512 686 480 706 400 L 712 378" stroke-width="33"/>'
    "</svg>"
)

#: Tokens lifted verbatim from the site's `@theme` block, light ramp in
#: ``:root`` and dark ramp under ``prefers-color-scheme``, so the two stay
#: comparable to their source by name as well as by value.
#:
#: The dot field (motif M5: a 1px dot on a 24px pitch, masked so it fades
#: before the edges) is the one texture the system allows, and it is what keeps
#: a single centred card from reading as an empty page.
_STYLE = """
:root {
  color-scheme: light;
  --paper: #f7f4ee; --surface: #fcfbf7; --sunken: #efece3;
  --ink: #211e18; --ink-muted: #565147; --ink-dim: #6c675c;
  --hairline: #e5e0d5; --hairline-strong: #d5cfc2;
  --success: #1e7b4e; --danger: #b23a31;
}
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --paper: #16130e; --surface: #1e1a14; --sunken: #0f0c08;
    --ink: #f1eee6; --ink-muted: #b5afa2; --ink-dim: #918b7d;
    --hairline: #2b2619; --hairline-strong: #3b3527;
    --success: #57c785; --danger: #ef8078;
  }
}
* { box-sizing: border-box; }
html { height: 100%; }
/* SAFE centring. `place-items: center` on a full-height grid overflows a
   too-tall card equally in both directions, and the half above the top cannot
   be scrolled to: measured card.top = -36px at 844x390 with a realistic
   provider error, losing the status rule and the brand header — the two
   elements that say what happened and who is saying it. `align-content` gives
   away free space only when there is some, so overflow can only ever go
   downward, where the scrollbar reaches. */
body {
  margin: 0;
  min-height: 100%;
  display: grid;
  align-content: center;
  padding: 24px;
  background: var(--paper);
  color: var(--ink);
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  -webkit-font-smoothing: antialiased;
}
/* Motif M5. Masked on three layers so it fades out rather than terminating
   on an edge, and hidden on small viewports where there is no room for it to
   read as texture instead of noise. */
.field {
  position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background-image: radial-gradient(circle at 1px 1px, var(--hairline-strong) 1px, transparent 0);
  background-size: 24px 24px;
  -webkit-mask-image:
    radial-gradient(120% 80% at 50% 42%, #000 20%, transparent 72%),
    linear-gradient(to right, transparent 0%, #000 12%, #000 88%, transparent 100%),
    linear-gradient(to bottom, transparent 0%, #000 10%, #000 90%, transparent 100%);
  mask-image:
    radial-gradient(120% 80% at 50% 42%, #000 20%, transparent 72%),
    linear-gradient(to right, transparent 0%, #000 12%, #000 88%, transparent 100%),
    linear-gradient(to bottom, transparent 0%, #000 10%, #000 90%, transparent 100%);
  -webkit-mask-composite: source-in;
  mask-composite: intersect;
}
@media (max-width: 520px) { .field { display: none; } }
/* A card, by the system's Card/panel reading: `--radius-md`, and NO shadow.
   Elevation here is the `--surface`-on-`--paper` step plus one hairline, and
   the dot field stopping dead at the card's edge separates it further. The
   heaviest shadow the system owns has no business on the page whose own
   comment says nothing floats. */
.card {
  position: relative; z-index: 1;
  width: 100%; max-width: 30rem;
  margin-inline: auto;
  background: var(--surface);
  border: 1px solid var(--hairline-strong);
  border-radius: 10px;
  overflow: hidden;
  /* No entrance animation, and that is a decision rather than an omission.
     A fade-in was tried and removed: an offscreen or throttled renderer never
     advances the animation's timeline, so the card sat at the first keyframe's
     `opacity: 0` and the page was BLANK — DOM correct, heading present, tests
     green, nothing on screen. Dropping the fill mode does not fix it, because
     a started-but-never-advanced animation still wins over the base style.
     The general rule this leaves behind: the one page whose entire job is to
     be read must not depend on motion to become legible. */
}
/* The page's single spend of colour: a 2px rule along the top edge, reading as
   a status light rather than a banner. 2px because the system has 1px borders
   and exactly one 2px exception; 3px is a weight it does not own. */
.card::before {
  content: ""; display: block; height: 2px;
  background: var(--tone, var(--hairline-strong));
}
.head {
  display: flex; align-items: center; gap: 12px;
  padding: 16px 24px;
  border-bottom: 1px solid var(--hairline);
  background: var(--sunken);
}
.mark { width: 24px; height: 24px; color: var(--ink); flex: none; }
.wordmark {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12px; line-height: 1.25; font-weight: 500; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--ink-muted);
}
.body { padding: 32px 24px 24px; }
h1 {
  margin: 0;
  font-family: ui-serif, "Iowan Old Style", Georgia, "Palatino Linotype",
    "URW Palladio L", serif;
  font-size: 2.5rem; line-height: 1.08; font-weight: 400; letter-spacing: -0.024em;
  color: var(--ink);
  text-wrap: balance;
}
.detail {
  margin: 12px 0 0;
  font-size: 1.0625rem; line-height: 1.65;
  color: var(--ink-muted);
  text-wrap: pretty;
}
/* Labelled troughs. Untinted on purpose: an accent pill with a button's radius
   reads as something to click, spends the accent the status rule needs, and
   leaks M2's chip grammar out of the band that owns it. `--sunken`'s
   documented role is "inset wells, code-adjacent troughs", which is what these
   are — and the label is what stops a screen reader announcing a bare URL with
   no idea what it is. */
.label {
  margin: 20px 0 0;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12px; line-height: 1.25; font-weight: 500; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--ink-dim);
}
.trough {
  display: inline-block; margin: 8px 0 0;
  padding: 4px 8px;
  border: 1px solid var(--hairline-strong);
  border-radius: 2px;                        /* --radius-xs */
  background: var(--sunken);
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.875rem; line-height: 1.5;
  color: var(--ink);
  /* `anywhere`, not `break-all`: break-all splits at the first opportunity that
     fits even when a better one is two characters later, which cuts a hostname
     mid-label on the one element whose job is to be read as a hostname. */
  overflow-wrap: anywhere;
}
.close {
  margin: 24px 0 0; padding-top: 16px;
  border-top: 1px solid var(--hairline);
  font-size: 0.875rem; line-height: 1.6;
  color: var(--ink-dim);
}
/* The system's narrow-viewport steps: a 20px page gutter below `sm`, and 20px
   card padding below `md`. LAST in the sheet on purpose — these override
   `body`, `.head` and `.body` at equal specificity, so source order is the
   whole mechanism, and putting them beside the rules they modify silently
   loses. (It did: the first attempt sat above `.body` and measured as a no-op.)
   Not cosmetic at 320px — the 16px they give back is the difference between
   the server URL fitting on one line and breaking two characters from its
   end, which splits the one string the trough exists to let you read. */
@media (max-width: 639.98px) { body { padding: 20px; } }
@media (max-width: 767.98px) {
  .head { padding: 16px 20px; }
  .body { padding: 32px 20px 20px; }
}
"""

#: Per-tone status rules. ``success`` takes the SEMANTIC token rather than the
#: brand accent — the failure half already uses ``--danger``, and "confirmed" is
#: what ``--color-success`` is for — which also leaves the brand accent unspent
#: on a page that has one message. ``neutral`` deliberately maps to a hairline:
#: a speculative browser fetch for ``/favicon.ico`` is not an event, and giving
#: it a coloured rule would teach the colour to mean nothing.
_TONE_VARS: dict[Tone, str] = {
    "success": "--tone: var(--success);",
    "danger": "--tone: var(--danger);",
    "neutral": "--tone: var(--hairline-strong);",
}

#: How much of a provider's ``error_description`` this page will render.
#:
#: The voice boundary gave the string its own trough; this gives it an extent.
#: The listener's 16 KiB request-head budget is the only other bound on it, and
#: a description that spends it renders a 7,461px card at 1417x1022 — pushing
#: OUR sentence, the one that says what to do next, some 7,000px below the fold
#: on a page with no navigation. 500 characters is roughly five lines in the
#: trough and keeps the whole card inside a desktop viewport with no scroll at
#: all; anything a provider needs to say beyond that belongs in their own UI.
#:
#: Truncated at the render boundary with a visible ellipsis rather than clipped
#: in CSS: `max-height` + `overflow` makes a nested scroll container that then
#: needs a `tabindex` to be keyboard-reachable, and `-webkit-line-clamp` hides
#: text with no sign that it did.
_MAX_PROVIDER_MESSAGE = 500


def _trough(label: str, value: str) -> str:
    return f'<p class="label">{html.escape(label)}</p><p class="trough">{html.escape(value)}</p>'


def render_callback_page(
    title: str,
    detail: str,
    *,
    tone: Tone = "neutral",
    server: str | None = None,
    provider: str | None = None,
    provider_message: str | None = None,
    closable: bool = True,
) -> str:
    """The full HTML document for one callback outcome.

    ``server`` is the MCP server the grant was for, shown in a labelled trough —
    omitted rather than faked when the caller does not know it. ``provider`` is
    the same trough for a model-provider login (Anthropic, OpenAI, Z.AI): the
    name of the party the user just authorized, labelled ``Provider`` so the
    page says *whose* login finished without pretending an MCP server was
    involved. The two are mutually exclusive at the call sites (one listener
    each), but nothing here needs to enforce that.

    ``provider_message`` is the third party's own ``error_description``, and it
    gets its own labelled trough rather than being spliced into ``detail``.
    That is a voice boundary, not decoration: the string is arbitrary text from
    a query parameter, rendered inside a card carrying our mark, and a sentence
    that begins in our voice and continues in theirs with no seam hands a
    hostile provider a paragraph of Local Operator-branded instruction. The
    trough says "this text is data, not us" — and it is also where a bare
    ``access_denied`` reads correctly, instead of appearing as English prose.

    ``closable`` adds the "you can close this tab" line, which is true of a
    finished grant and a lie on a page the flow is still waiting past.
    """
    blocks = ""
    if server:
        blocks += _trough("MCP server", server)
    if provider:
        blocks += _trough("Provider", provider)
    # Strip BEFORE the gate. `error_description=%20%20%20` is truthy and
    # reaches here from the wire, and gating on the raw argument then rendering
    # the stripped text produces a labelled empty box — the exact thing the
    # absent-message case exists to avoid.
    trimmed = (provider_message or "").strip()
    if trimmed:
        if len(trimmed) > _MAX_PROVIDER_MESSAGE:
            trimmed = trimmed[: _MAX_PROVIDER_MESSAGE - 1].rstrip() + "…"
        blocks += _trough("Provider response", trimmed)
    close = (
        '<p class="close">You can close this tab and return to Local Operator.</p>'
        if closable
        else ""
    )
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<meta name='robots' content='noindex'>"
        f"<title>{html.escape(title)} · Local Operator</title>"
        f"<style>{_STYLE}</style></head>"
        f"<body style='{_TONE_VARS[tone]}'>"
        "<div class='field' aria-hidden='true'></div>"
        "<main class='card'>"
        f"<header class='head'>{_LOGO_MARK}<span class='wordmark'>Local Operator</span></header>"
        f"<div class='body'><h1>{html.escape(title)}</h1>"
        f"<p class='detail'>{html.escape(detail)}</p>{blocks}{close}</div>"
        "</main></body></html>"
    )


def callback_response(
    title: str,
    detail: str,
    *,
    tone: Tone = "neutral",
    server: str | None = None,
    provider: str | None = None,
    provider_message: str | None = None,
    closable: bool = True,
    status: str = "200 OK",
) -> bytes:
    """:func:`render_callback_page`, wrapped in a complete HTTP/1.1 response.

    ``Connection: close`` and an explicit ``Content-Length`` because the
    listener answers exactly one request per connection and then goes away;
    without both, a keep-alive browser holds the socket open and the flow's
    teardown has one more thing to wait for.
    """
    body = render_callback_page(
        title,
        detail,
        tone=tone,
        server=server,
        provider=provider,
        provider_message=provider_message,
        closable=closable,
    ).encode()
    head = (
        f"HTTP/1.1 {status}\r\n"
        "Content-Type: text/html; charset=utf-8\r\n"
        f"Content-Length: {len(body)}\r\n"
        "Cache-Control: no-store\r\n"
        "Connection: close\r\n\r\n"
    ).encode()
    return head + body
