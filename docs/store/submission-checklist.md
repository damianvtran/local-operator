# Chrome Web Store submission checklist

Run this against the final release candidate. Do not submit from a development
build or from copy that no longer matches the manifest.

## 0. Resolve the launch assumptions

- [ ] Choose and publish a monitored support/privacy email for Radient, Inc.
      This is not supplied by the design docs and remains an explicit
      placeholder in `privacy-policy.md` and `permissions.md`.
- [ ] Confirm the privacy-policy route. This package proposes
      `https://local-operator.com/browser-extension/privacy`; the source docs
      do not reserve a URL.
- [ ] Confirm the final extension version and minimum supported Chrome version.
      The design depends on Chrome 116+ WebSocket service-worker lifetime
      behavior, though older Chromium forks can reconnect on alarms.
- [ ] Decide the store-icon background color with design. The supplied icon
      sources are transparent white/black glyphs and need a controlled square
      background to work on both light and dark store surfaces.
- [ ] Confirm whether Radient, Inc. will identify as a “trader” for EEA consumer
      law fields, and supply any required business address/D-U-N-S details.
      This is a legal/account-owner decision, not defined by the product design.

## 1. Developer account and publisher identity

- [ ] Sign in to the Chrome Web Store Developer Dashboard with a company-owned,
      durable Google account controlled by **Radient, Inc.**, not a personal
      throwaway account.
- [ ] Confirm two-step verification is enabled for the Google account.
- [ ] Confirm the developer registration is complete and the one-time **US $5
      registration fee** shows as paid. Official Google documentation states
      that a one-time fee is required; the task brief specifies $5. If the
      dashboard shows a localized/different amount, record and pay the amount
      it requires rather than forcing this historical figure.
- [ ] Set publisher/developer display name to **Radient, Inc.**
- [ ] Verify the developer contact email and turn on item and account
      notifications. Use a mailbox monitored after launch.
- [ ] Complete any identity, physical-address, phone, trader, or payment-profile
      verification requested for the company account.
- [ ] Verify ownership of `local-operator.com` in the Google account used by the
      publisher, then select it as the extension's **Official URL**. This is
      what lets the store show the official-site/verified-publisher treatment;
      the display name alone does not.
- [ ] Open the eventual store listing in a signed-out browser and confirm the
      publisher renders as Radient, Inc. with the official URL. **Assumption:**
      “verified publisher” in the task means domain-backed official publisher
      status; Google's exact badge language can change.

## 2. Freeze and inspect the package

- [ ] Build the extension from the release commit using the repository's
      documented command: `node build.mjs --zip` from `extension/`.
- [ ] Record the release commit SHA, extension version, `PROTO_VERSION`, zip
      filename, zip SHA-256, build command, Node version, and pnpm lockfile
      commit in the release record.
- [ ] Unzip into a temporary directory and inspect the artifact itself:
  - [ ] `manifest_version` is 3.
  - [ ] name is `Local Operator`; no dead “Patch” name remains.
  - [ ] version matches the intended store release.
  - [ ] permissions exactly match the dashboard declarations.
  - [ ] host permissions are no broader than implemented need.
  - [ ] all scripts, CSS, icons, HTML, and source maps intended for submission
        are present and local.
  - [ ] no secrets, test fixtures, personal paths, development URLs, or remote
        script references are present.
  - [ ] no `eval`, remotely fetched executable code, arbitrary CDP Runtime
        evaluation, or generic “execute JavaScript” bridge command exists.
- [ ] Confirm the Python bridge and extension enforce the same protocol version.
- [ ] Run extension gates: frozen pnpm install, TypeScript check, production
      build, generated-protocol check, and applicable tests.
- [ ] Load the exact unzipped artifact through `chrome://extensions` in a clean
      Chrome profile and complete the manual matrix from design §9.5: pairing,
      all eight actions, persistent login across app-session restart, site
      allow and deny, redirect gate, browser/worker failure and reconnect,
      concurrent sessions, and invalid/unauthorized local RPC requests.

## 3. Host public policy and support pages

- [ ] Publish `privacy-policy.md` as rendered HTML at the chosen permanent
      `https://local-operator.com/...` URL. It must be publicly accessible
      without login, region gating, JavaScript-only authentication, or a local
      app.
- [ ] Replace the effective-date and contact-email placeholders first.
- [ ] Make the page title and body explicitly name the **Local Operator browser
      extension** and **Radient, Inc.**
- [ ] Confirm a fresh private-window request returns HTTP 200 and the page is
      readable on mobile and desktop.
- [ ] Link the policy from a discoverable Local Operator website page as well as
      the CWS dashboard.
- [ ] Confirm the GitHub Issues support URL is monitored, or replace it with the
      public support route chosen by Radient, Inc.

## 4. Produce and verify listing assets

- [x] Export a 128 × 128 store icon from the source/crop documented in
      `assets.md`; inspect it at actual size on light and dark backgrounds.
      → `docs/store/assets/store-icon-128.png`, verified on white and dark cards.
- [x] Capture five real 1280 × 800 release-build screenshots following the
      `assets.md` storyboard: connected, pairing, site prompt, live
      terminal/browser work, and reconnecting.
      → `docs/store/assets/screenshot-{1..5}-*.png`. **Storyboard deviation:**
      the fifth panel ships as the options/allowed-sites view
      (`everything stays on your machine`) rather than the reconnecting state,
      per the task brief's requested five states (connected, allow, pairing,
      agent-driving, options). The disconnected/reconnecting frame
      (`popup-disconnected.png`) remains available if the reconnecting story is
      preferred for slot 5.
- [x] Export the 440 × 280 small promo tile.
      → `docs/store/assets/promo-small-440x280.png`.
- [x] Export the optional 1400 × 560 marquee tile now if a designer is
      available; it avoids a separate merchandising scramble.
      → `docs/store/assets/promo-marquee-1400x560.png`.
- [x] Verify no personal data or still-valid pairing code appears in any image.
      Sources are demo frames (`example.com` origins, empty pairing field,
      `127.0.0.1` test page); no account avatar, email, or live code present.
- [ ] Confirm the agent-work screenshot shows a real completed path and
      Chrome's debugger disclosure rather than hiding it. **Open:** screenshot 4
      shows the completed path (`Page Two` reached, popup `Driving:` line) but
      the supplied capture omits Chrome's debugger banner. Re-capture with the
      banner visible before upload; see the caveat in `assets.md`.
- [x] Keep original captures and editable compositions with the release record.
      → `build_assets.py` regenerates every PNG from the committed source frames.

## 5. Create the dashboard item

- [ ] Click **New item** and upload the release zip.
- [ ] Save the generated item/extension ID. Pairing pins this ID on the local
      bridge, so production testing and documentation must use it.
- [ ] If the store-assigned ID differs from a development ID, rebuild/configure
      only through the implementation's supported production-ID mechanism;
      never hard-code around origin checks ad hoc.
- [ ] Upload the 128 px icon, five screenshots, small promo tile, and optional
      marquee.

## 6. Complete the Store Listing tab

Use `listing.md` as the copy source.

- [ ] Title: **Local Operator**.
- [ ] Short description: paste and recheck the dashboard's character count is
      at or below 132.
- [ ] Long description: paste as plain store text, inspect line breaks, and
      remove Markdown markers if the dashboard does not render Markdown.
- [ ] Category: **Tools**. Both direct peers checked on 2026-08-25, Claude and
      ChatGPT's agent extension, use Tools rather than Workflow & Planning.
- [ ] Language: **English**.
- [ ] Official URL: verified `https://local-operator.com` property.
- [ ] Homepage and support URLs are public and correct.
- [ ] Privacy policy URL is the hosted page from step 3.
- [ ] No mature content; no in-app purchases; free.
- [ ] Preview desktop/mobile listing. Check that the first screenshot, title,
      and first two description lines explain the benefit without expansion.

## 7. Complete the Privacy Practices tab

Use `permissions.md` as the source and reconcile it with the final manifest.

- [ ] Paste the single-purpose statement exactly:

      > Connects the Local Operator app on this computer to this browser so its agent can browse on the user's behalf.

- [ ] Paste a specific justification for every final permission:
      `debugger`, `tabs`, `scripting`, `storage`, `alarms`, `webNavigation`, and
      host permissions. Remove any row absent from the final manifest; add any
      permission introduced by implementation.
- [ ] Declare **no remote code** and paste the explanation from
      `permissions.md`.
- [ ] Disclose website content because the extension reads page text,
      accessibility data, field results, and screenshots.
- [ ] Work through the current definitions for web history, user activity,
      authentication information, and potentially sensitive website content.
      When Google's form is broader than our colloquial use of “collect,”
      disclose conservatively and explain that processing is task-scoped and
      local. Do not create a mismatch between dashboard and policy to make the
      list shorter.
- [ ] Certify Limited Use statements: no sale, no unrelated use/transfer, and
      no creditworthiness/lending use.
- [ ] Enter the hosted privacy-policy URL.
- [ ] If a reviewer instructions/test-credentials field is present, paste the
      pairing run-through from `permissions.md` and provide a monitored support
      contact. No account credentials are required; the reviewer does need the
      local app and pairing code.

## 8. Extra scrutiny for `debugger` + broad hosts

This combination receives slower and stricter review. Treat reviewer clarity
as a release requirement, not copy polish.

- [ ] The listing says what `debugger` does: screenshots, accessibility
      snapshots, trusted clicks, and typing in one delegated tab.
- [ ] The justification explains why content scripts are insufficient.
- [ ] It says the connection is a daemon on `127.0.0.1`, not a remote operator.
- [ ] It says agent messages are typed **data**, not code; all executable code
      ships in the extension.
- [ ] It states that the extension cannot evaluate arbitrary model-provided
      JavaScript. Confirm that is true in the built source.
- [ ] It explains default-deny per-origin prompts, including redirect targets.
- [ ] It explains HTTP(S)-only runtime checks despite broad host declaration.
- [ ] It explains that only the agent-owned tab is inspected and Chrome's
      debugger banner stays visible while attached.
- [ ] Provide the short pairing/demo procedure so a reviewer can reproduce the
      single purpose without guessing.
- [ ] Upload a short pairing-flow demo if the dashboard/support exchange permits
      attachments. The design explicitly calls for one.

This addresses likely review questions directly: why debugger is necessary,
who issues commands, whether code is remote, what tabs/sites can be accessed,
and what control remains with the user.

## 9. Distribution and submission

- [ ] Visibility/distribution: **Public**.
- [ ] Pricing: **Free**; no paid features in the extension.
- [ ] Select all supported regions unless Radient, Inc. has a legal reason to
      exclude one. **Assumption:** the source docs specify public/free but no
      regional restriction.
- [ ] Do not request enterprise-only or unlisted distribution.
- [ ] Re-open every saved tab. Dashboard validation can reset fields when the
      uploaded package changes.
- [ ] Have a second person compare the zip manifest, listing, permissions copy,
      privacy policy, and screenshots for contradictions.
- [ ] Submit for review early. Keep the documented sideload path available
      (`chrome://extensions` → Developer mode) while review is pending.
- [ ] Save submission timestamp, version, item ID, status screenshots, and any
      warnings shown by the dashboard.

## 10. Expected review timeline and responses

Google's published historical benchmark says most submissions completed in
less than 24 hours and more than 90% within three days, with developer-support
contact recommended after three weeks. That benchmark dates to early 2021 and
is not an SLA. For this first release, plan **several business days to multiple
weeks**, because `debugger` plus broad site access is explicitly
high-scrutiny. Do not promise a launch date until approval.

- [ ] Monitor the publisher mailbox and dashboard daily.
- [ ] Answer reviewer questions with the same narrow facts in this package; do
      not broaden capability while explaining it.
- [ ] If rejected, preserve the exact rejection text and submission artifact.
      Fix the source issue, copy mismatch, or missing evidence. Do not merely
      reword a truthful permission need into vagueness.
- [ ] If pending past three weeks, use Chrome Web Store One Stop Support and
      include item ID, submission time, and a concise pairing/test procedure.
- [ ] Any code change after submission requires a new build, complete manifest
      reconciliation, regression pass, and likely resubmission. Never upload a
      “small fix” zip without repeating these checks.

## 11. Post-approval

- [ ] Open the public listing while signed out and install it into a clean
      supported Chrome profile.
- [ ] Run the production path against the released Local Operator app: install
      bridge, pair, approve a site, open/read/click/type/screenshot, close, and
      reconnect after browser restart.
- [ ] Verify extension version, publisher, official site, privacy link,
      screenshots, category, language, and free/public status on the live page.
- [ ] Test installation from the same Chrome Web Store URL in Edge, Brave, and
      Arc. The design supports any Chromium browser that installs Chrome
      extensions, but the store page itself should not claim platform behavior
      that was not exercised.
- [ ] Add the canonical Chrome Web Store listing URL to the repository README
      and browser documentation. Put it next to installation instructions, not
      only in release notes.
- [ ] Add the listing URL to `local-operator.com` download/browser-extension
      documentation and verify it is clickable.
- [ ] Retain a sideload section for users whose browser/profile cannot access
      the store, clearly marked as the advanced fallback.
- [ ] Record the approved version, approval timestamp, item ID, listing URL,
      artifact SHA-256, source commit, and protocol version in the extension
      release record.
- [ ] Establish the update runbook: build from a release commit, inspect zip,
      upload, reconcile newly requested permissions, refresh screenshots/copy
      when behavior changes, submit, monitor, and run the production smoke test.
- [ ] Watch support reports for redirect prompt fatigue, MV3 reconnect latency
      on older Chromium forks, protocol mismatch states, and unexpected
      debugger detachments. These are the rollout risks named in the design.
