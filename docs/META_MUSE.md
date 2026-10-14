# Unofficial Meta Muse Code subscription sign-in

`lop login meta-muse` (or `/login meta-muse` in the app) approves a device on
Meta's account site, then exchanges that account token for a **Muse Code
inference key**. The desktop uses the existing `/v1/auth/login` operation
contract: approval URL, user code, expiry, cancellation and final status. It
needs no new frontend login API and never opens a browser from the backend.

**This is unofficial and unsupported by Meta outside its own CLI.** An active
subscription and a successful key exchange are not a promise about future
billing, model entitlement or coverage of every request. Review Meta's current
terms before use. Contributor routes have separate contribution terms and are
not offered by this integration. It does not impersonate the official CLI's
telemetry or silently enroll an account into Contributor.

**Privacy:** official Muse documentation describes Contributor/default settings
that can use prompts, code and context for training. Disable Contributor in
Meta/Muse settings **before proprietary or customer use**. Choosing a
non-Contributor model ID in lop does not prove those account settings changed;
lop neither changes nor verifies them, and sends no speculative
`x-tbh-contributor-mode` override.

## Protocol and verification boundaries

The independently implemented device protocol follows the public client ID,
OIDC device authorization/token routes and `/muse-code/key` exchange observed
by OpenCodex and oh-my-pi. First-party model/reasoning docs establish the
Responses wire and effort vocabulary. The account token and inference key are
**different credentials**; only the minted inference key is sent to
`/v1/responses`.

`GET /muse-code/models` is the subscription-specific, authenticated catalogue.
Native configuration lives under `data[].metadata["muse-code"]`; its visibility,
text-output modality, tool/image support, context/output limits and effort
variants control discovery. Full is offered only if the actual
`muse-wonder-full-1.0` row is present and not hidden for that account, never by
inventing a public API alias. Hidden/image-only/Contributor routes are excluded.
Catalogue presence is not a promise that every request will be accepted.

An inference stream must emit a valid `response.subscription_usage` receipt
for **that request** before its drained end is reported successful. The receipt
is a subscription-meter observation, not an independent invoice check or a
promise of no overage. Integer percentages can stay unchanged after a tiny
request: that is inconclusive numerical debit, not evidence a call was free.
Do not spend extra inference traffic merely to force a visible usage change.

Recorded testing evidence belongs on the PR, not in this repository. Protocol
fixtures exercise safety boundaries but are not live billing/entitlement proof;
a live meter and a human dashboard attribution check are distinct sources and
must be reported as such.

## Catalogue freshness and existing conversations

A user-opened live model picker refreshes Muse's authenticated listing even
when its successful cache is young. Existing fetch leases, failure backoff and
stale-cache fallback still apply; this is fresh-on-open, not a push feed or a
quota/inference poll. Network-free first paint uses only a metadata-identified
account document (or the controller's last actual selected listing scope), not
an arbitrary sweep of other accounts. Native names, family/release date,
modalities, output limits, capability denials and effort variants survive the
cache and additive desktop/canonical detail rows. Missing facts remain unknown.
Selected-account metadata scopes session limits and effort; Codex's optional
maximum-budget setting and account headers do not apply to Muse.

An installed update does not change code already loaded by a running
conversation. A healthy newly sealed Muse sign-in cannot be read by an older
owner that never implemented that provider. New owners advertise native
provider support; a stale legacy owner without positive support evidence
refuses a new builtin choice **before changing the conversation**, with
`runtime_upgrade_needed` rather than a missing-sign-in diagnosis. This does not
restrict new model IDs under an affirmed provider or custom compatible hosts.
Use a new conversation, or let the existing runtime reach its normal idle
update boundary and resume the same conversation. Busy work is never forcibly
restarted; no plaintext credential migration or new sign-in is the repair.
The composer access notice follows the effective serving route while preserving
the selected primary, and a saved but unavailable sign-in is distinguished
from one that is absent. Device approval displays the complete backend privacy,
unsupported-integration and billing disclosure as inert text; copying the code
remains the explicit browser-opening gesture.

## Safety boundaries

- This provider has **no `META_API_KEY`/`MODEL_API_KEY` environment leg**, no
  legacy PAYG-key namespace and no generic paste-key login. A key shaped like
  `LLM|…` alone is not subscription provenance, so manual/static import is not
  offered without a verified entitlement path.
- Missing/inactive subscription state, unknown payment state or payment-required
  mint responses fail closed even if Meta supplies a usable key. No PAYG route
  is selected. HTTP/stream errors cannot cause automatic billable fallback.
  A completed/incomplete response without a valid meter receipt for **that
  request** fails closed after draining the stream; previous/concurrent receipts
  cannot satisfy it. A meter arriving after `response.completed` is still read.
  This cannot retroactively guarantee the first request's invoice; it prevents
  silent API-only success or additional automatic traffic.
- The inference key is sealed in the encrypted provider-secret namespace at
  the owning configuration root. `auth.db` holds only a restricted metadata
  allowlist and an opaque reference; it never holds the raw inference/account
  tokens. Keys are account-isolated. Failed metadata writes remove the newly
  sealed orphan, not a previously working key. Logout removes only that
  account's sealed key. This store is not a vault: same-user code able to run
  the secret CLI can still retrieve credentials.
- Account tokens are discarded after mint. There is no automatic refresh or
  remint; a rejected static key requires an explicit new sign-in. Cancellation
  reaps in-flight requests. Each auth request has a 20-second bound; grant TTL
  and polling intervals are bounded, and redirects are not followed.
- Response bodies, exception reprs, tokens and device codes never appear in
  auth diagnostics. The user code and verified `auth.meta.com` approval URL
  are the only approval material shown. Passive allowance receipts retain
  only finite numeric usage/reset/duration fields and an observation timestamp;
  opaque tier IDs and other response fields are dropped.
- Meter snapshots are **observations**, not a polling service. There is no
  verified read-only quota endpoint, and the app never creates a chargeable
  inference solely to refresh quota. Raw percentages above 100 remain raw;
  any eventual display may clamp its bar, not the underlying evidence.
- Muse uses public Responses, `store:false`, encrypted reasoning replay and
  automatic function tools. No Codex headers, prompt-cache retention, custom
  tools or unsupported `tool_choice` are sent. A no-tools call removes tools;
  required-tool mode is explicitly refused. Wire-safe names are mapped back
  to original harness tool names without collisions.

## Sources

- [Meta coding-agent integration](https://dev.meta.ai/docs/coding-agents)
- [Meta reasoning guide](https://dev.meta.ai/docs/reasoning)
- [OpenCodex issue #3191 and implementation history](https://github.com/lidge-jun/opencodex/issues/3191)
- [OpenCodex subscription-usage observations](https://github.com/lidge-jun/opencodex/blob/dev/src/providers/muse-subscription-usage.ts)

These are behavioral/protocol references, not copied implementation. Official
Muse SDK/MSP serve invokes a nested agent rather than supplying a raw inference
stream to this harness, and is not substituted silently for this provider.
