# Chrome Web Store listing copy

Store-facing product: **Local Operator** (the browser extension). Voice per
local-operator.com and docs/design/branding-bridge.md: plain words, short
declaratives, benefit first, no hype. The "Patch" name from the round-1
branding exploration is dead; the microcopy *patterns* there (state lines,
prompt phrasing) are the reference, not the name.

## Title

> Local Operator

14 characters (limit 45). Matches the store-naming precedent the design doc
cites (Claude and ChatGPT list under their product names, not capability
names).

## Short description

> Let Local Operator browse in your real browser with your real logins. The browser connection stays on your machine.

115 characters (limit 132).

## Long description

> **Your browser, with an agent in it.**
>
> This extension connects the free, open-source Local Operator app to the
> browser you already use — Chrome, Edge, Arc, or Brave. The agent can open
> pages, read them, click, type, and take screenshots, in your real browser
> with your real logins. Sign in to a site by hand and the agent carries on
> from there.
>
> **The browser connection stays on your machine.**
>
> The extension talks to a small bridge service running on your own computer,
> over the local loopback connection only. No cloud browser, no remote
> session, no analytics. The extension sends browser data only to your own
> Local Operator app and never sends anything directly to us or another
> remote service. Conversation text and browser results the agent needs may be
> processed by the AI model you chose in Local Operator; choose a local model
> if you want that processing to stay on-device too.
>
> **You decide which sites.**
>
> The agent works in one tab of its own — never the tab you are using. It can
> only open sites you have allowed. The first time it wants a new one, the
> extension asks, and you pick a scope: all pages on this domain, only this
> site, just this once, or deny. While the agent drives the tab, the browser
> shows its own banner on that tab, so you always know.
>
> **Paired to your app and nothing else.**
>
> A one-time pairing code, shown in your own terminal and typed into the
> extension, ties this browser to your copy of the Local Operator app. No
> account, no sign-up. Unpair any time from the extension.
>
> **Free, like the rest.**
>
> The extension is free and pairs with the free, MIT-licensed Local Operator
> app. Supported by Radient, Inc.
>
> ---
>
> Requires the Local Operator app running on the same computer
> (https://local-operator.com). Works on any Chromium browser that installs
> Chrome extensions.
>
> To drive the page — take screenshots, read the accessibility tree, and send
> real clicks and keystrokes — the extension uses Chrome's debugger API on
> the one tab you delegated. Chrome shows a notice on that tab whenever this
> is active. That notice is the point: you can always see when the agent has
> the tab.

Notes on the description:
- The debugger paragraph is deliberate: CWS reviewers read the listing, and a
  listing that owns the `debugger` permission up front reviews better than
  one that hides it (design doc §11 "Store review").
- No em-dash avoidance rule applies here (that is the release-notes format);
  the site copy itself uses em-dashes, so the listing may too.

## Category

**Tools** (under Productivity).

Checked against the direct peers on 2026-08-25: both the Claude extension
(chromewebstore…/fcoeoabgfenejglbffodgkkbkcdhcgfn) and the ChatGPT agent
extension (chromewebstore…/hehggadaopoacecdllhhajmbjkdcmajg) list under
**Tools**, as do the adjacent bridge-style tools (Browser MCP, Playwright
Extension). Workflow & Planning is dominated by PDF/meeting/clipper
utilities, not agent bridges. Tools is where users comparing agent
extensions will look.

## Language

English (`en`). Single language at launch; the app itself ships English-only.

## Search keywords

CWS has no separate keyword field — search indexes the title and
description — so these terms are woven into the copy above (verify they
survive edits):

- AI agent, browser agent, agent for your browser
- Local Operator, local AI, on-device, open source
- browser automation, drive the browser, fill forms, click, type, screenshot
- Chrome / Edge / Arc / Brave
- private, local browser connection, no cloud browser, no analytics, your logins

## Fields the dashboard also asks for

| Field | Value |
|---|---|
| Official website | https://local-operator.com (verify the domain in the developer dashboard to get the "created by the owner of the listed website" badge) |
| Support URL | https://github.com/damianvtran/local-operator/issues |
| Privacy policy URL | https://local-operator.com/browser-extension/privacy — **assumption**: exact path not fixed anywhere yet; see privacy-policy.md preamble |
| Mature content | No |
| In-app purchases | None |
| Price | Free |
