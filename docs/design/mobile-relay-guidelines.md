# Mobile relay: design and UX guidelines

**Canonical guide for new and changed relay UI.** Scope: `local_operator/mobile/web`.
Use this with the shared theme/brand system, not as a second token source. The
[redesign record](mobile-relay-redesign.md) preserves historical measurements and
rationale; its snapshots are not a statement of today's UI. When changing an
existing contract, update this guide and supply the measured reason in the PR.

## Status: contracts, evidence and goals are different

The sections below are requirements for changes, not a claim that every existing
screen already complies. Existing primitives implement parts of them; a PR must
still exercise the affected real flow. In particular, **first-editable startup
and durable optimistic drafts below are required future behavior, not guarantees
shipped by this chrome-polish change**. Do not describe mock/component coverage,
static replay, an old screenshot or an intended interaction as current live proof.

## Ground, hierarchy and type

- Conversation content and the next required decision are primary. Navigation,
  diagnostics and completed tool detail stay secondary and progressively disclosed.
- Separate regions by the existing `canvas` → `surface` → `elevated` ground ladder
  and spacing; use `sunken` for a well. No redundant card/panel outlines. Focus
  outlines, necessary input boundaries and semantic state cues are not panel borders.
- Use semantic theme roles (`text-ink`, `text-ink-muted`, `text-ink-dim`, state and
  ground roles), never feature-local palette values or opacity on whole text rows.
  Test both Local Operator light/dark themes; preserve other palettes' identities.
- Use the existing type ramp. Prose is readable body text; labels/metadata are the
  quieter meta role; monospace is for code and compact numeric/technical facts.
  Keep contrast at the existing checker floors (normally 4.5:1 text, 3:1 meaningful
  UI graphics). Do not solve density by shrinking primary text or touch controls.
- A long title truncates in its own `min-w-0` flex slot, not by pushing actions
  outside the viewport. Reserve stable slots for changing status and counts.

## Brand and icons

- The visible Local Operator identity in home/navigation chrome is the official
  **white circle with the dark motif**, unchanged in both light and dark themes.
  No emoji, font-symbol logo, CSS reconstruction, tint, inversion or substitute
  white-on-dark motif. The artwork itself is an intentional fixed-color brand
  asset; this is not permission to hardcode component colors. The explicit
  conversation/navigation identity request is a narrow exception to the native
  brand kit's no-filled-disc rule, not a new rule for all iconography.
- Relay asset: `local_operator/mobile/web/src/assets/brand-icon.png`, a byte-for-byte
  copy of the desktop repository's
  [`resources/icon-small.png`](https://github.com/damianvtran/local-operator-ui/blob/19bf873610d10268e7d0bcc1dfad9a8901d53a21/resources/icon-small.png).
  SHA-256: `e438b0430a45a6b42c4e1a49ac8b52c5e5fdefe0955251d94abfbcc0217f7113`.
  The upstream [branding guide](https://github.com/damianvtran/local-operator-ui/blob/main/docs/branding.md)
  points to `damianvtran/local-operator-design` as the brand home of record.
- Reuse `BrandMark`: 24 px, restrained, in the existing conversations and drawer
  header rows. It is decorative beside a named heading, not an extra focus target.
  Inline the artwork so an identity-proxy login redirect cannot break an image
  fetch. Do not replace unrelated login/app-icon assets as incidental cleanup.
- Use the shared `ChromeIcon` SVG for menu/search/close: **24 px visible icon box**,
  consistent stroke and optical weight, inside **at least 44 × 44 px independent
  tap targets**. Enlarging only the button does not enlarge the glyph. Verify the
  actual path/ink extent as well as SVG and target rectangles in a rendered frame.
- Icons inside named buttons are `aria-hidden`, unfocusable and non-interactive.
  Preserve the button's localized accessible name, expanded state, controls
  relationship and focus behavior. Mirror directional meaning in RTL, not neutral
  search/menu symbols or the brand artwork.
- Keep the home header at its existing 52 px plus safe-area inset and the drawer
  title within its existing row. A new branding strip must not consume content
  height. Search may temporarily use the title slot; do not squeeze the editor to
  preserve a decorative mark.

## Composer and startup

- Align the composer with its conversation column. Use balanced optical insets,
  consistent control gaps and stable action slots; do not stack independent padding
  layers until one edge looks heavier. Measure content bounds and control centers,
  not only declared CSS padding. Keep model/cwd and required actions usable in
  narrow and short viewports without hiding text below the keyboard.
- **First-editable startup — requirement, not yet shipped here:** opening a new
  conversation must expose an editable draft immediately. Session creation, model
  lists, directory discovery or reconnects must not hold the user's typing behind
  a remote spinner. Represent pending setup separately from the editable content.
- **Durable optimistic draft safety — requirement, not yet shipped here:** persist
  draft content and its stable client identity before asynchronous setup/send;
  preserve it through route changes, reload, failure, retry and session-expiry /
  reauthentication. Recovery must belong to the same authenticated identity, never
  leak to another account. Associate it with the eventual session without duplicate
  ownership. Clear it after authoritative acknowledgement and durable transfer,
  not when a request merely starts; explicit discard/sign-out still follows the
  auth/private-storage deletion policy. Authentication-preserving recovery needs
  its own security-reviewed implementation; this document does not change that policy.
- A pending or failed start/send must say what happened, retain the user's work,
  and offer a safe retry. Repeated taps and delayed acknowledgements must not
  create duplicate sessions or sends. These invariants need real request/side-effect
  evidence before being promoted from requirements to implemented guarantees.

## Responsive layout and overflow

- Support 320/360/390 px phone columns and short landscape viewports, including
  safe areas, keyboard-visible states, zoom/large text and long localized labels.
  The document itself must not scroll horizontally. Use shrinkable flex children,
  wrapping and explicit overflow ownership rather than clipping required controls.
- Lists and navigation are vertically discoverable. Never hide destinations or
  filter choices behind an unmarked sideways strip. Preserve selected/filter state
  when an overflow menu is needed.
- Wide tables/code may own a bounded local horizontal scroller with an observable
  cue; retain table semantics, headings and cell relationships. Reflow simple
  key/value content when it improves reading. Do not make the whole page pan or
  silently discard columns to make an overflow assertion pass.
- Fades belong to the actual scroll owner and reflect whether more content exists.
  Keep scrollbars/edge cues discoverable; never fade or mask actionable error text,
  focus rings or the only indication of selection. Avoid double fades at nested
  scroll boundaries.

## Drawers, sheets and disclosure

- Reuse the existing modal layer: named dialog, background inertness, focus trap,
  keyboard Escape, visible close target and focus return to the invoker. Keep the
  opening press from dismissing the surface. Provide both scroll and dismissal
  access in short/landscape viewports; closing must not discard a draft.
- Respect logical `start/end` alignment, safe-area padding and localized names.
  Do not add unlabeled icon buttons, competing traps or a second modal framework.
- Progressive disclosure must be reversible: expansion restores the exact rows
  hidden, without duplicating ownership/counts or losing long prompts. A collapsed
  failed activity must expose failure both visually and in its accessible name.
- Keep related levels visually subordinate without shrinking hit targets. Avoid
  unnecessary stacked summaries when one disclosure can safely own the same state;
  do not consolidate stateful levels without proving restoration and live updates.

## Motion and reading position

- Animate state, not layout churn. Respect reduced motion; no unsolicited shimmer
  or movement on settled content. An arrival, loading completion or changed count
  must not move the action the reader was about to press.
- Preserve a deliberate disclosure's pressed header and a reader's scroll anchor
  through the next render, not merely the first frame. Browser clamping at a real
  scroll boundary is a distinct case: disclose its actual displacement rather than
  calling every interaction zero-motion. Do not fabricate blank space to hide it.
- Do not drag a reader back to the tail when they scrolled away. Genuine streaming
  transitions require genuine producer/event evidence; prop changes or a synthetic
  backwards `final` transition are not proof of the live lifecycle.

## Required state matrix and review evidence

For each affected surface, cover the relevant cells; name unavailable cells rather
than marking them passed:

| Axis | Minimum discriminating cells |
|---|---|
| Size | 320, 360, 390 portrait; short landscape; safe-area/keyboard where relevant |
| Theme/content | Native light/dark; long title/labels/query; large text; LTR and RTL |
| Data | Loading, empty, populated, error/retry; reconnect/stale state when affected |
| Interaction | Closed/open drawer or sheet; search/open/close/clear/focus; keyboard/Escape; repeated activation |
| Persistence | Pending/acknowledged/failed/retried draft/start/send, **when that behavior is implemented** |

Before requesting review:

1. Write the discriminating regression first and observe the failure; run affected
   tests/checks serially and bounded. Reuse dependencies; let CI supply the terminal
   whole-suite coverage rather than duplicating it locally.
2. Preserve a source-identified BEFORE frame. Build the changed source once, then
   exercise the real rendered path through the harness browser. No substitute
   browser engine if that host is unavailable; report the exact blocker.
3. Capture and **view** before/after stills, plus consecutive frames for motion.
   Record viewport, document width, glyph/SVG/target rectangles, chrome height and
   scroll-owner geometry. Pixel evidence and geometry must agree.
4. Check accessible names/states and actual focus/keyboard behavior, not just DOM
   attributes. Record real outputs and failures, not an inferred pass from tests.
5. Obtain independent code/QA review and design review for visuals, UX review for
   changed flows. Post per-finding dispositions and retain the actual evidence
   paths. State which guarantees are implemented, which are requirements, and
   which remain unproven. Follow `AGENTS.md` for the complete repository gates.
