# Chrome Web Store submission checklist

Run this against the final release candidate. Do not submit from a development
build or from copy that no longer matches the manifest.

## Live session status (2026-09-06) — where we actually are

The extension is **published**. The first-publication gates described in the
previous revision of this section (Radient, Inc. business verification and the
EEA trader declaration) were cleared as of 2026-09-02. They are not routine
re-checks — but they can reopen if the account owner, legal entity, or
declaration details change, so re-verify on any such change rather than assuming
them settled forever.

- **v0.1.7 is live** on the Chrome Web Store at 100% deployment, item
  `omibaecbjdhgbbcedbnnnmjpmopfheof`, promoted 2026-09-04 (run 33926643637). It
  supersedes v0.1.0, and also v0.1.5, which was the pending submission when this
  section was last written.
- **v0.1.8 is submitted and `PENDING_REVIEW`** at 100% deployment, from source
  commit `ff9b16b5` (PR #672). See the release record in
  `docs/store/release-record.md`.
- **Automated publishing has now carried two real releases.** v0.1.7 was the
  first, submitted and published end to end through the workflows (runs
  33815585846 and 33926643637); v0.1.8 followed through `chrome-web-store.yml`
  (run 34000911184). It is the intended path for every subsequent release. See
  "Automated updates after the first public release" below.
- **Known platform limit, still true:** Chrome forbids the debugger API on the
  Web Store domains ("The extensions gallery cannot be scripted"), so the
  extension cannot auto-fill the developer console — a human drives the console
  pages. Documented in `guide://browser`. This is why a release that introduces
  a new permission needs a human in the dashboard.

- **`main` carries v0.1.8, which is the submitted version.** The tree and the
  pending submission agree for now; the manifest tracks the code rather than the
  review queue, so any further extension change on `main` moves past 0.1.8 again.
  Nothing auto-fires when it does — `chrome-web-store.yml` is
  `workflow_dispatch` with an explicit `ref` and `version` behind a protected
  environment. A release-record entry is owed only when a version actually
  ships.

**Pick-up point:** watch for the v0.1.8 review decision, then run the
post-approval steps in section 11 — including appending the approval timestamp
to the v0.1.8 entry in `release-record.md`, and promoting the staged release.
Sections 0–9 describe *first* publication and are retained as the reference for
listing copy and asset requirements; re-walk them only when the listing,
permissions, or privacy policy change.

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

## Permission handoff (`tabGroups`, added in 0.1.4)

- [x] **Done for the 0.1.5 upload (2026-09-02).** The `tabGroups` justification
      from `permissions.md` was entered in the dashboard and 0.1.5 was submitted
      against the eight-permission declaration, not the original seven.

**Standing rule for future releases:** a package that adds a permission cannot
go out through the automated workflow alone. The Chrome Web Store API cannot set
permission justifications, and Chrome blocks scripting the dashboard, so a human
must paste the rationale from `permissions.md` before submitting. Diff the built
`extension/dist/manifest.json` against the previously published version's
permission list on every release to catch this; when they differ, upload by hand.

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
- [ ] Refresh the bundled Public Suffix List and confirm it is current:
      `node scripts/gen-psl.mjs` then `node scripts/gen-psl.mjs --check`.
      Its own step rather than a clause in the bullet above, because this is
      the only place drift is actually gated: the CI step is advisory
      (`continue-on-error`, since it fetches publicsuffix.org and an upstream
      blip must not fail an unrelated PR). A suffix registered after
      `PSL_GENERATED_AT` is treated as a registrable domain, so a stale list
      lets one approval cover every name beneath it.
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
      shows the completed path (the reached page in a browser frame, and the
      popup's "Connected." state over that same `127.0.0.1` test-page URL) but
      no asset shows Chrome's debugger banner. The reached page is typeset by
      `build_assets.py` rather than captured, so restoring the banner is a
      builder change, not a re-capture; see the caveat in `assets.md`.
- [x] Keep original captures and editable compositions with the release record.
      → `build_assets.py` regenerates every PNG from the committed source frames.

## 5. Finish the permanent dashboard item

The permanent item/extension ID is
**`omibaecbjdhgbbcedbnnnmjpmopfheof`**. Pairing pins this identity on the local
bridge; do not create a replacement item or hard-code around origin checks.

- [ ] Open that existing item in the Developer Dashboard; API v2 cannot create
      items or change their visibility.
- [ ] Upload the first release zip manually if the item has never been published.
      Chrome Web Store requires the first publication, and the first publication
      after any visibility change, to be completed in the dashboard before API
      publishing can preserve that visibility.
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

- [ ] Visibility/distribution: **Public**. Set and publish this manually for the
      first release; API v2 deliberately cannot change visibility and will only
      preserve the dashboard's already-published setting.
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

### Automated updates after the first public release

**Status (2026-09-06): configured, verified end to end, and proven on two real
releases.** Both protected environments, the workload identity provider, the
service-account IAM binding, and the store's authorization of that service
account were confirmed working on 2026-09-02; the evidence is in
`release-record.md`. **v0.1.7 was the first version actually released through
the workflows** — submitted 2026-09-03 (run 33815585846) and published
2026-09-04 (run 33926643637), no dashboard step in either half. **v0.1.8**
followed on 2026-09-06 (run 34000911184), validating and submitting with
`STAGED_PUBLISH`. Every release before v0.1.7 went out by hand. The workflow
fails closed before uploading anything, and the two earlier staged attempts both
stopped well short of the store: run 33766746870 never started a single step,
sitting 4h29m awaiting environment approval before its pending deployment failed
(annotation: `The deployment was rejected or didn't satisfy other protection
rules`) seconds after #585 removed the required-reviewer gate it was queued
behind; run 33793517588 got as far as `Verify protected environment
configuration` and exited 22 on a GitHub API 403, with the authenticate and
upload steps skipped — the failure #589 then fixed. Neither reached the Chrome
Web Store, so a surprise there is recoverable.

The workflows use Chrome Web Store API v2 directly with `curl` and exchange a
GitHub OIDC token for a short-lived service-account access token through
`google-github-actions/auth@v3` (pinned by full commit SHA, like every action in
the token-bearing workflows). Do not add a service-account JSON key, OAuth
client secret, refresh token, or third-party upload action, and do not bump the
pinned action SHAs without reviewing the new revision.

Both workflows **fail closed**: before authenticating they verify through the
GitHub API that their environment exists with a custom deployment branch
policy allowing exactly `main`, and that every release variable is defined
**on the environment itself**. Repository- or organization-scoped copies are
rejected so nobody can bypass environment protection by defining the same
names at a broader scope.

The environments previously also carried at least one required reviewer, and
that human gate was removed by deliberate operator decision on 2026-09-03 so
a store release completes end to end automatically. Do not re-add it without
reopening that decision. The protections that remain are the real boundary:
the WIF attribute condition pinning the numeric repository ID and
`refs/heads/main` (so only reviewed code on main can mint a token), the
exact-main deployment branch policy, the environment-scoped variables, and
each workflow's own `git merge-base --is-ancestor HEAD origin/main` check.
The launch decision now rides on the reviewed-merge-to-main process plus that
WIF ref pin.

The environment-scoped variable check is enforced by a **job topology**, not by
an API call, and removing either half silently disarms it. Each store workflow
runs a `preflight` job that deliberately has **no `environment:` key**: outside
the environment a correctly-scoped variable resolves to the empty string, while
a repository- or organization-scoped one is visible to every job and so resolves
non-empty and fails the release. The deploying job (`stage`/`promote`) then
`needs: preflight` and asserts the same names resolve non-empty inside the
environment. Do not delete the `preflight` job, give it an `environment:`, add
an `if:`/`continue-on-error:` to either job, or point its checkout at
`inputs.ref` — each of those removes the only guard against a repo-scoped
variable steering a release. `extension/tests/release.test.mjs` asserts all of
this, so a change that breaks the topology fails the extension suite.

The reason it is a topology rather than an API call: `GET
/repos/{o}/{r}/environments/{env}/variables` requires the `environments=read`
permission, which a workflow `GITHUB_TOKEN` cannot be granted at all (a
`permissions: write-all` job still receives 403 and its granted-permission list
contains no `Environments` entry). See the header of
`extension/scripts/verify-release-environment.sh`.

Define these environment variables (identifiers, not secrets) on **each** of
`chrome-web-store` and `chrome-web-store-production`:

- `CWS_PUBLISHER_ID` — Publisher ID from Developer Dashboard → Account.
- `CWS_EXTENSION_ID` — `omibaecbjdhgbbcedbnnnmjpmopfheof`.
- `GCP_WIF_PROVIDER` — full Workload Identity Provider resource name.
- `CWS_SERVICE_ACCOUNT` — service-account email authorized in the Chrome Web
  Store Developer Dashboard.

Enable Chrome Web Store API in the GCP project and add the service account
under Developer Dashboard → Account. Create the WIF provider with issuer
`https://token.actions.githubusercontent.com/`, this attribute mapping:

```
google.subject=assertion.sub,attribute.repository_id=assertion.repository_id,attribute.ref=assertion.ref
```

and exactly this attribute condition. It pins the **numeric repository ID**
(`922327641` for `damianvtran/local-operator`; renames and repo-name reuse
cannot forge it, unlike the owner/name string) and the `main` ref, so a token
minted from any other repository or branch cannot impersonate the release:

```
assertion.repository_id == "922327641" && assertion.ref == "refs/heads/main"
```

Grant the pool principal `roles/iam.workloadIdentityUser` on the service
account scoped with the same `attribute.repository_id` value. Do **not**
configure required reviewers on either GitHub environment — the human approval
gate was removed by operator decision on 2026-09-03 (see above); each
environment keeps its custom deployment branch policy allowing exactly `main`
(a policy of type `branch`, never a tag named `main`, and never a glob such as
`main*`) and its environment-scoped variables. Define the release variables on
the environment only — defining any of them at repository or organization scope
fails the `preflight` job by design. Dispatch both workflows from `main` —
the WIF ref condition and each workflow's own main-ancestry check reject
anything else.

1. Run **Chrome Web Store staged release** manually with a commit/tag already on
   `main` and the exact manifest version. With no environment approval gate it
   runs immediately: it installs frozen dependencies, runs typecheck/tests,
   builds exactly the source-map-free zip, verifies manifest/package/input
   versions and an explicit zip allowlist, uploads it, and submits with
   `STAGED_PUBLISH` at 100% deployment. If the
   store reports the upload as asynchronous, the run fails closed — the API's
   global upload status cannot be bound to this zip — and is safe to re-run
   once processing finishes.
2. Wait for Chrome review and verify `fetchStatus` reports that exact version as
   `STAGED`. Review the approved package/listing before launch.
3. Run **Promote staged Chrome Web Store release** with that exact version. The
   dispatch itself is the explicit launch decision — the production
   environment no longer pauses for a human approver (removed by operator
   decision on 2026-09-03) — and the run refuses non-staged, mismatched, or
   partially-deployed versions and verifies the result is `PUBLISHED` at 100%
   deployment.

The existing GitHub Release asset behavior remains unchanged and independent of
store publication.

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
      release record, `docs/store/release-record.md`.
- [ ] Establish the update runbook: build from a release commit, inspect zip,
      upload, reconcile newly requested permissions, refresh screenshots/copy
      when behavior changes, submit, monitor, and run the production smoke test.
- [ ] Watch support reports for redirect prompt fatigue, MV3 reconnect latency
      on older Chromium forks, protocol mismatch states, and unexpected
      debugger detachments. These are the rollout risks named in the design.
