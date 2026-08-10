"""The page the browser lands on at the end of an MCP OAuth grant.

This is the only web surface local-operator serves, and it is served at the
worst possible moment: the user has just handed a third party permission on
their behalf and is looking for confirmation that the thing they authorized
actually received it. A bare ``<h2>`` on a white page reads like a server
error even when it says "Authorized", so this page is drawn in the product's
own design language — the Local Operator marketing site's system
(``local-operator-site/docs/design-language.md``): warm paper, one green,
hairlines instead of shadows, Fraunces-style serif display over a humanist
sans, and a dark ramp for a user whose OS is dark.

Three constraints shape the implementation:

- **Zero network.** No font CDN, no stylesheet, no image request. The tab must
  render identically on a laptop that is offline or behind a proxy that has not
  been authorized yet, and a page that phones out at the end of an OAuth flow
  is the wrong thing to hand someone thinking about permissions. So: one
  document, inline ``<style>``, inline SVG, and the design system's own
  declared fallback stacks (``ui-serif, Georgia`` / ``ui-sans-serif, system-ui``
  / ``ui-monospace``) instead of the three webfonts the site self-hosts.
- **One file, no build step.** The site is React + Tailwind v4; none of that can
  run here. The tokens below are copied from that system's ``@theme`` block
  with their names kept, so a change there is greppable here.
- **Both ramps, always.** ``prefers-color-scheme`` swaps the ramp via CSS
  variables exactly as the site's ``.dark`` block does. Every pair below is
  from the system's contrast tables: body ink is AAA on its ground in both
  ramps, and the dimmest text used (``ink-dim``) is 5.43:1 light, 5.11:1 dark.

The tone argument is not decoration. Success and failure of an authorization
are the one thing this page exists to distinguish, and the system's rule is
that semantic colour is only ever spent on real semantics — so ``success``
lights the mark and the rule in the brand green, ``danger`` in the danger red,
and ``neutral`` (the 404 a browser's speculative fetch earns) spends nothing.
"""

from __future__ import annotations

import html
from typing import Literal

Tone = Literal["success", "danger", "neutral"]

#: The Local Operator mark, traced in `local-operator-site`'s `logo-mark.tsx`.
#: Inlined rather than fetched (see the module docstring) and stroked in
#: ``currentColor`` so the tone class alone colours it.
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
  --paper: #f7f4ee; --surface: #fcfbf7; --sunken: #efece3;
  --ink: #211e18; --ink-muted: #565147; --ink-dim: #6c675c;
  --hairline: #e5e0d5; --hairline-strong: #d5cfc2;
  --accent: #177b45; --accent-wash: #e7f1e8; --accent-border: #47795b;
  --danger: #b23a31; --danger-wash: #f7e7e4; --danger-border: #96544c;
  --shadow-frame: 0 1px 2px rgb(20 17 12 / 0.06), 0 24px 48px -24px rgb(20 17 12 / 0.28);
}
@media (prefers-color-scheme: dark) {
  :root {
    --paper: #16130e; --surface: #1e1a14; --sunken: #0f0c08;
    --ink: #f1eee6; --ink-muted: #b5afa2; --ink-dim: #918b7d;
    --hairline: #2b2619; --hairline-strong: #3b3527;
    --accent: #38c96a; --accent-wash: #16281d; --accent-border: #4a8160;
    --danger: #ef8078; --danger-wash: #2e1b18; --danger-border: #9e5a51;
    --shadow-frame: 0 1px 2px rgb(0 0 0 / 0.4), 0 24px 48px -24px rgb(0 0 0 / 0.6);
  }
}
* { box-sizing: border-box; }
html, body { height: 100%; }
body {
  margin: 0;
  display: grid;
  place-items: center;
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
.card {
  position: relative; z-index: 1;
  width: 100%; max-width: 30rem;
  background: var(--surface);
  border: 1px solid var(--hairline-strong);
  border-radius: 16px;
  box-shadow: var(--shadow-frame);
  overflow: hidden;
  /* No entrance animation, and that is a decision rather than an omission.
     A fade-in was tried and removed: an offscreen or throttled renderer never
     advances the animation's timeline, so the card sat at the first keyframe's
     `opacity: 0` and the page was BLANK — DOM correct, heading present, tests
     green, nothing on screen. Dropping the fill mode does not fix it, because
     a started-but-never-advanced animation still wins over the base style.
     The general rule this leaves behind: the one page whose entire job is to
     be read must not depend on motion to become legible. It also happens to be
     what the design language asks for — nothing on this page floats. */
}
/* The header rule is the only place the tone is spent as a fill: a hairline
   the width of the card, tinted, reading as a status light along the top edge
   rather than as a banner. */
.card::before {
  content: ""; display: block; height: 3px;
  background: var(--tone, var(--hairline-strong));
}
.head {
  display: flex; align-items: center; gap: 10px;
  padding: 18px 28px;
  border-bottom: 1px solid var(--hairline);
  background: var(--sunken);
}
.mark { width: 22px; height: 22px; color: var(--tone, var(--ink)); flex: none; }
.wordmark {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12px; line-height: 1.25; font-weight: 500; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--ink-dim);
}
.body { padding: 32px 28px 28px; }
h1 {
  margin: 0;
  font-family: ui-serif, Georgia, "Times New Roman", serif;
  font-size: 2.5rem; line-height: 1.08; font-weight: 400; letter-spacing: -0.024em;
  color: var(--ink);
  text-wrap: balance;
}
.detail {
  margin: 14px 0 0;
  font-size: 1.0625rem; line-height: 1.65;
  color: var(--ink-muted);
  max-width: 42ch;
  text-wrap: pretty;
}
/* The server the grant was for. A user with several MCP servers configured
   has no other way to tell which tab belongs to which authorization. */
.server {
  display: inline-block; margin: 20px 0 0;
  padding: 6px 10px;
  border: 1px solid var(--accent-border);
  border-radius: 6px;
  background: var(--accent-wash);
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.875rem; line-height: 1.5;
  color: var(--ink);
  word-break: break-all;
}
.server.danger { border-color: var(--danger-border); background: var(--danger-wash); }
.close {
  margin: 24px 0 0; padding-top: 18px;
  border-top: 1px solid var(--hairline);
  font-size: 0.875rem; line-height: 1.6;
  color: var(--ink-dim);
}
"""

#: Per-tone accents. ``neutral`` deliberately maps to a hairline: a speculative
#: browser fetch for ``/favicon.ico`` is not an event, and giving it a coloured
#: rule would teach the colour to mean nothing.
_TONE_VARS: dict[Tone, str] = {
    "success": "--tone: var(--accent);",
    "danger": "--tone: var(--danger);",
    "neutral": "--tone: var(--hairline-strong);",
}


def render_callback_page(
    title: str,
    detail: str,
    *,
    tone: Tone = "neutral",
    server: str | None = None,
    closable: bool = True,
) -> str:
    """The full HTML document for one callback outcome.

    ``server`` is the MCP server the grant was for, shown as a mono chip —
    omitted rather than faked when the caller does not know it. ``closable``
    adds the "you can close this tab" line, which is true of a finished grant
    and a lie on a page the flow is still waiting past.
    """
    chip = ""
    if server:
        chip_class = "server danger" if tone == "danger" else "server"
        chip = f'<p class="{chip_class}">{html.escape(server)}</p>'
    close = (
        '<p class="close">You can close this tab and return to your terminal.</p>'
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
        f"<p class='detail'>{html.escape(detail)}</p>{chip}{close}</div>"
        "</main></body></html>"
    )


def callback_response(
    title: str,
    detail: str,
    *,
    tone: Tone = "neutral",
    server: str | None = None,
    closable: bool = True,
    status: str = "200 OK",
) -> bytes:
    """:func:`render_callback_page`, wrapped in a complete HTTP/1.1 response.

    ``Connection: close`` and an explicit ``Content-Length`` because the
    listener answers exactly one request per connection and then goes away;
    without both, a keep-alive browser holds the socket open and the flow's
    teardown has one more thing to wait for.
    """
    body = render_callback_page(title, detail, tone=tone, server=server, closable=closable).encode()
    head = (
        f"HTTP/1.1 {status}\r\n"
        "Content-Type: text/html; charset=utf-8\r\n"
        f"Content-Length: {len(body)}\r\n"
        "Cache-Control: no-store\r\n"
        "Connection: close\r\n\r\n"
    ).encode()
    return head + body
