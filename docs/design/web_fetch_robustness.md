# Design: `web_fetch` robustness — transient failures, bot-blocks, and honest diagnostics

Status: **implemented**. The manager ratified §12 as: `blocked_retry` default on
(1), the 0.6 stall fraction ships fixed with no knob (2), no card hint row and
no UI parity PR (3), the empty-`str(exc)` fix folded in (4), ship (5).

Three things the implementation had to settle that this document did not, each
recorded here rather than only in the PR so the next reader of the design sees
the shipped contract:

1. **Enrichment probes get a bounded SLICE of the deadline** (25 %), not the
   whole budget. §6.2 made the probes share the call's deadline, which is
   correct, but §3.3 did not say what stops a stalling `.md` probe from spending
   the budget the real fetch — and its escalation — needs. Measured during
   implementation: a 3 s fetch of a silent origin spent 1.8 s on the probe and
   had nothing left to escalate with. `_ENRICHMENT_BUDGET_FRACTION` is the fix.
2. **The `blocked` lead drops the "(The body below is the error response…)"
   note.** §5.2 replaces the challenge body with our own statement, so that note
   would describe the wrong thing. Other non-2xx classes keep it unchanged.
3. **§6.6 case 5's literal assertion is wrong and was implemented as its
   intent.** It asks that an unmatched 403's `describe()` "does not contain the
   word 'bot'", but §2.2's own prescribed wording for that case is "No anti-bot
   vendor signature was found … rather than bot protection" — which contains the
   word precisely in order to *deny* the claim. The test asserts the claim is
   absent (`"blocked by"` never appears, the denial does), which is the property
   §2.2 is actually about.

A fourth, smaller one: `describe()` pre-wraps its prose to the TUI card's row
width. The card paints one body row per LINE and clips the overflow with an
ellipsis (pre-existing behaviour, visible on the `Fetched:` row in both the
before and after frames), so an unwrapped 150-column sentence would have lost
the name of the escalation tool off the right edge.

Companion to `docs/design/web_fetch.md`, which remains the contract for the
pipeline, SSRF policy, caching and card. This document extends it; it does not
replace it. Where the two could be read as disagreeing, §11 says which wins.

Grounded against the tree at `origin/main` (`c938fead6`). Every `file:line`
citation and every measurement below was taken in this worktree
(`~/workspace/repos/lo-webfetch-robust`, its own `.venv`) on 2026-09-12. Numbers
are measured, not estimated; where a number is a judgement call it says so.

---

## 1. The problem, as the code actually has it

The operator asked a shopping question in the desktop app. Every `web_fetch` in
that turn came back `is_error=True`, the model-facing body was the origin's
Akamai block page presented under a warning lead, the `browser` escalation was
unavailable (extension not paired), and the agent then spent ~40 tool calls
probing URL shapes with no working next step.

The UI rendered that faithfully — each red cross matches an `is_error=True`
result in the transcript. The defect is in the harness tool, and it is three
defects, not one.

### 1.1 There are no retries anywhere in the fetch path

`WebFetchService._follow` (`local_operator/web_fetch/service.py:510-551`) makes
exactly one `_stream_once` call per hop, and `_stream_once`
(`service.py:597-637`) converts any `httpx.TimeoutException` or `httpx.HTTPError`
into a terminal `FetchError`. A read timeout, a connection reset, a mid-body EOF,
a 500, a 502, a 503 or a 429 therefore fails the whole call on the first try.
`run_fetch` turns that into an error result (`tool.py:442-443`).

This is not theoretical on real hosts. Measured in this worktree, with the
shipped UA and the shipped 20 s timeout:

```
https://www.canadiantire.ca/en/pdp/certified-6-outlet-...-0527211p.html
  -> httpx.ReadTimeout after 20.21s   (and after 5.02s with timeout=5)
```

The same host answers a browser-shaped request in **0.07-0.25 s with a 403**.
So the stall is not a slow origin: it is the anti-bot layer black-holing a
request it has already decided to refuse. Today that costs the agent the full
timeout budget and yields no information.

### 1.2 There is exactly one request identity, and it is a blocked one on real sites

`USER_AGENT` (`service.py:65`) is the only request header the engine sets
(`service.py:539`). No `Accept`, no `Accept-Language`, no `Sec-Fetch-*`. Measured
flip on a site the operator plausibly reads, 3/3 reproductions:

```
https://medium.com/   lop UA      -> 403, 5,507 bytes, cf-mitigated: challenge,
                                     body <title>Just a moment...</title>
https://medium.com/   browser-shaped profile
                                  -> 200, ~53,067 bytes, the real page
```

Confirmed to still hold **through the SSRF-pinned path** (`_pin_request` to
`162.159.153.4`, `Host`/SNI preserved): 403 with the lop profile, 200 with the
browser-shaped one. So the fix rides the existing guarded path; it does not need
a second client.

Controls that do **not** flip, and which are the reason this must not become
site-specific logic: `walmart.ca`, `reddit.com`, `nytimes.com`, `amazon.ca`,
`instagram.com`, `tiktok.com`, `zillow.com` all return 200 with the plain lop UA.

### 1.3 A blocked response is presented as content, and a stalled one as nothing

`_header_line` (`tool.py:156-181`) leads a non-2xx with an unmissable warning and
then prints the rendered body. For a block page that body is the challenge
markup — 5.5 KB of Cloudflare interstitial, or Akamai's "Access Denied /
Reference #18.…". Measured through the real CLI:

```
$ .venv/bin/python -m local_operator.cli fetch test "https://medium.com/"
error: ⚠ HTTP 403 Forbidden — this is an error/block page, not page content. https://medium.com/
markdownify · text/html; charset=utf-8 · cache miss
(The body below is the error response, not the requested page.)

Just a moment...
```

The agent is told *that* it failed and shown a body it cannot use. It is not told
**why** (bot protection, not a missing page, not an auth wall), **whether
retrying could help**, or **what to do next**. Nothing in the result names the
escalation, so the agent invents one — which is exactly the ~40-call flail.

The stalled case is worse. Measured through the real CLI:

```
$ .venv/bin/python -m local_operator.cli fetch test "https://www.canadiantire.ca/...0527211p.html"
   (20.5 s later)
error: timed out fetching 'https://www.canadiantire.ca/...0527211p.html': 
```

`httpx.ReadTimeout.__str__()` is the **empty string**, so `service.py:634`'s
f-string produces a message that ends in `': '` with nothing after it. The reader
cannot tell a connect failure from a read stall from a slow page, and the message
looks like a truncation bug.

### 1.4 What the code already does right, and must keep doing

- SSRF: `validate_public_url` runs per hop (`service.py:530-532`) and the socket
  is pinned to the vetted IP with `Host`/SNI preserved (`_pin_request`,
  `service.py:185-214`). TLS verification is never disabled.
- Only 2xx is cached (`tool.py:478-479`), so a transient outage is never replayed
  for the TTL.
- `web_fetch` and `read <url>` share one engine (`run_fetch`, `tool.py:334`).
- Requests are GET-only — `client.stream("GET", …)` at `service.py:618-620` is the
  single request site in the package (verified by grep: no `.post`, `.put`, or
  generic `.request` call exists in `local_operator/web_fetch/`).

Every change below preserves all four.

---

## 2. Failure taxonomy and detectors

One classifier, `classify_failure`, living in a new
`local_operator/web_fetch/failure.py`. It returns a small frozen dataclass
(`FetchFailure(kind, retryable, detail, marker, vendor)`) and is the single place
any of these strings live, so the engine, the tool text and the tests cannot
drift.

| Class | Detector | Retry? |
|---|---|---|
| `transport` | `httpx.ConnectError`, `httpx.ConnectTimeout`, `httpx.ReadError`, `httpx.WriteError`, `httpx.RemoteProtocolError`, `httpx.PoolTimeout` | yes |
| `stall` | `httpx.ReadTimeout` / `httpx.WriteTimeout` (headers or body never arrived) | yes, once, on a shortened budget (§3.3) |
| `server` | status in {500, 502, 503, 504} | yes |
| `ratelimit` | status 429 | yes, honouring `Retry-After` (§3.4) |
| `blocked` | status in {403, 401-with-anti-bot-marker, 503-with-challenge-marker} **and** a marker below | not a retry — a **profile escalation**, once (§4) |
| `client` | any other 4xx (404, 410, 451, …) | no |
| `ok` | 2xx/3xx | n/a |

### 2.1 The anti-bot markers, and why these are the stable ones

Matched case-insensitively against **response headers first, body second**.
Headers are the stable signal; body strings are the fallback for vendors that do
not brand their headers. All of the following were captured live in this
worktree today, not copied from memory:

**Cloudflare** — header `cf-mitigated: challenge` (captured verbatim on
`https://medium.com/`; Cloudflare sets it specifically to let non-browser clients
distinguish a challenge from a real 403, which is why it is the most stable
marker we have). Body fallbacks: `<title>Just a moment...</title>`,
`Attention Required! | Cloudflare`, `challenges.cloudflare.com` in a CSP or
script src. Secondary corroboration: `server: cloudflare` plus `cf-ray`.

**Akamai** — header `server: AkamaiGHost` (captured verbatim on
`shoppersdrugmart.ca` and `canadiantire.ca`). Body fallbacks: the
`<TITLE>Access Denied</TITLE>` / `<H1>Access Denied</H1>` pair, the literal
`errors.edgesuite.net`, and `Reference #` followed by a dotted id. The
`Reference #` token is the piece a human support agent asks for, which is why it
is worth extracting (§5).

**DataDome** — headers `x-datadome`, `x-dd-b`, or a `datadome=` cookie in
`set-cookie`; body `captcha-delivery.com`.

**PerimeterX / HUMAN** — body `_pxhd` / `px-captcha` / `perimeterx.net`;
`set-cookie` names beginning `_px`.

**Generic** — `server: Imperva`/`X-Iinfo` (Imperva/Incapsula), and a 403 whose
body is under 2 KB and contains both `access denied` and no `<body>` text beyond
a title, which catches the long tail of WAF stock pages.

Markers are matched against the **first 8 KB** of the decoded body only. A
challenge page is always small; scanning a 5 MB body for substrings is wasted
work, and a marker appearing 3 MB into a real article is a false positive we do
not want.

### 2.2 What an *unmatched* 403 does — and this is the honesty-critical part

An unmatched 403 is treated as `blocked` **for the retry decision** (it earns the
one profile-escalation attempt, because a plain-403 WAF with no branding is
common and the escalation is cheap) but is **never labelled with a vendor**. Its
model-facing text says:

> `403 Forbidden — the origin refused this request. No anti-bot vendor signature
> was found, so this may be an access restriction rather than bot protection.`

versus the matched case:

> `403 Forbidden — blocked by Cloudflare bot protection (challenge).`

Claiming "bot protection" on a 403 that is really "you are not a subscriber"
would send the agent to `browser` when it should ask the user for credentials.
The distinction costs one boolean and buys the agent a correct next step.

### 2.3 The blackhole case

`stall` exists as a separate class from `transport` because its remedy is
different. Measured on `canadiantire.ca`: the lop profile never gets headers back
(ReadTimeout at 20 s), while a browser-shaped profile gets a 403 in 0.07-0.25 s.
A stall is therefore frequently a *silent* block, and the right response is the
same profile escalation §4 describes — on a **shortened** budget, because the
whole point is not to spend another 20 s learning nothing.

---

## 3. Retry policy — exact numbers

### 3.1 Attempts

**Maximum 3 network attempts per hop, and at most one profile escalation per
fetch.** Concretely, per redirect hop:

- attempt 1: default profile
- attempt 2: default profile, after backoff — only for `transport`, `server`,
  `ratelimit`
- attempt 3: default profile, after backoff — same classes only

and, orthogonally, **one** browser-shaped attempt (§4) when the *final* outcome
of the hop is `blocked` or `stall`. A fetch therefore issues at most 4 requests
to one hop, and a `blocked` result costs exactly 2.

Why 3 and not 5: the value of a retry is concentrated in the first one. A single
retry covers the overwhelming majority of one-off resets and 502s from a
rolling deploy; a third covers a short 503 window; a fourth is mostly waiting.
Against that, this tool runs inside a live turn the user is watching, and every
attempt is latency the user sees. 3 is the point where added coverage stops
paying for added wall-clock.

### 3.2 Backoff shape and jitter

`delay(n) = min(BASE * 2**(n-1), CAP) * (1 + random.uniform(-J, +J))` with
`BASE = 0.5 s`, `CAP = 4.0 s`, `J = 0.25` (full ±25 % jitter, same family as the
TUI's sidebar backoff at `tui/app.py:1618`).

So: retry 1 waits 0.5 s ± 0.125, retry 2 waits 1.0 s ± 0.25. Total added *sleep*
in the worst non-rate-limited case is ~1.9 s.

Jitter exists because a turn can issue several `web_fetch` calls in parallel (the
tool description explicitly encourages it) and they will frequently hit the same
origin. Un-jittered backoff synchronises those retries into a burst, which is
precisely what a struggling origin does not need.

### 3.3 Staying inside the per-call `timeout_seconds` budget

**This is a hard invariant: a `web_fetch` with `timeout_seconds=T` must not take
materially longer than T, retries or not.** Today `timeout` is handed to the
client as the per-request timeout (`service.py:538`), so a retried fetch would
multiply it. It becomes a **deadline for the whole call** instead:

- `_fetch_owned` computes `deadline = monotonic() + timeout` once.
- Each attempt gets `remaining = deadline - monotonic()`, and its httpx timeout
  is `httpx.Timeout(connect=min(5.0, remaining), read=remaining,
  write=remaining, pool=min(5.0, remaining))`. (Verified: `client.stream()`
  accepts a per-request `timeout`, and `httpx.Timeout` takes granular
  connect/read/write/pool values.)
- Before sleeping a backoff, check it fits: if `delay > remaining - 0.25 s`, stop
  retrying and return the failure now rather than sleeping into the deadline.
- **Attempt budgeting for the stall class:** attempt 1 of a hop is capped at
  `min(remaining, 0.6 * timeout)`. That is what converts the canadiantire case
  from "20 s then nothing" into "12 s then a fast browser-shaped probe that
  returns a real 403 in ~0.25 s". The 0.6 split is a judgement call; the
  evidence that would tune it is the distribution of time-to-first-byte on
  genuinely slow-but-working origins, which we do not have. 0.6 is chosen because
  a page that has sent no headers in 12 s is overwhelmingly more likely to be
  black-holed than slow. It applies **only when there is a fallback attempt left
  to spend** — the last attempt always gets the full remaining budget, so a
  genuinely slow origin is never penalised on its final try.

Worst-case added latency versus today, stated plainly:

| Scenario | Today | After | Delta |
|---|---|---|---|
| 200 first try | t | t | **0** (same one request) |
| 404 / other non-retryable 4xx | t | t | **0** |
| block detected (medium.com) | 0.06 s | ~0.3 s | +~0.25 s (one extra request) |
| stall (canadiantire, T=20) | 20.2 s | ~12.3 s | **−8 s** |
| persistent 500 | t | t + ~1.9 s sleep + 2 requests | bounded by T |
| any case, hard ceiling | T | T | **0** — the deadline is the bound |

The last row is the one that matters: the deadline never moves, so no
configuration of retries can make a call exceed its stated timeout.

### 3.4 `Retry-After`

Honoured on 429 and on 503 (both define it). Parsed as delta-seconds; the
HTTP-date form is also parsed (`email.utils.parsedate_to_datetime`) because CDNs
do emit it. Then:

- If `retry_after <= min(remaining - 0.25, 10.0)`: sleep exactly that (plus
  jitter), then retry. Honouring the origin's own number is the polite and
  effective behaviour.
- If it is larger: **do not sleep**. Return the failure immediately, with the
  value surfaced in the text (`the origin asked us to wait 120s`) and in
  `details`. Blocking a live turn for two minutes to obey a header is worse for
  the user than telling the agent to come back later, and the agent can act on
  the number.

A `Retry-After` of 0 or an unparseable value falls back to the normal backoff.

### 3.5 Interruptibility

Unchanged in shape and must stay so. `_fetch_or_abort` (`tool.py:291-317`) races
the whole `service.fetch()` coroutine against the abort signal, so it already
covers everything inside it. The backoff sleeps are plain `await
asyncio.sleep(...)` inside that coroutine, so a cancellation lands on them
immediately and the `finally` block still reaps the task. **No change to
`_fetch_or_abort`**, and a test asserts that an abort during a backoff sleep
returns promptly rather than after the sleep.

### 3.6 What the user-visible duration does

The card's duration is wall-clock for the tool call, so a retried call honestly
shows a longer duration — which is correct and needs no change. What changes is
that the *reason* is legible: the attempt count rides in the header meta line
(§5) and in `details["attempts"]`, so a 2.4 s fetch that used to look like a slow
network now reads `… · 3 attempts`.

### 3.7 Only GET is retried, and that is trivially true here

Confirmed by inspection: `client.stream("GET", …)` at `service.py:618` is the
only request the package issues; there is no POST/PUT path to guard. Retrying a
GET is safe by HTTP semantics (idempotent), and there is no code path through
which a caller could make this tool issue a non-idempotent request. If a future
change adds one, the retry predicate must gate on method — a comment at the retry
site will say so.

---

## 4. The blocked-retry profile

### 4.1 The default identity does not change

I agree with the manager's lean and state the reason affirmatively: **the honest
`local-operator/web_fetch` UA remains the default on every first attempt.**

- It is truthful, and truthfulness is the norm for an automated client. A site
  that wants to allow or deny us should be able to.
- The controls in §1.2 show it is not generally blocked — seven major sites
  serve it fine. Changing the default would be paying a lie on ~100 % of
  requests to fix a minority.
- `robots.txt`-conscious operators and site owners can identify us today; a
  Chrome-by-default client removes that.

The browser-shaped profile is therefore **the price of a refusal, not the
default posture** — paid only after the origin has already refused the honest
request.

### 4.2 Exact headers on the escalated attempt

Sent as a per-request header override on the same pinned, validated request
(verified: per-request headers merge over client headers and override them
key-by-key):

```
User-Agent:      Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36
                 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36
Accept:          text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,
                 image/webp,*/*;q=0.8
Accept-Language: en-US,en;q=0.9
Sec-Fetch-Site:  none
Sec-Fetch-Mode:  navigate
Sec-Fetch-User:  ?1
Sec-Fetch-Dest:  document
Upgrade-Insecure-Requests: 1
```

This is exactly the set measured to flip medium.com. Deliberately **not**
included: `sec-ch-ua*` client hints (they are Chrome-version-coupled and were
measured not to matter), cookies (the no-cookie policy in
`web_search/io.py:25-30` is a deliberate boundary and stays), and any
`Referer` (inventing a referrer is a fabrication, not a profile).

The `Host` header and `sni_hostname` extension from `_pin_request` are merged
**last** so the escalated attempt cannot accidentally unpin itself.

The UA string is a module constant next to `USER_AGENT` with a comment saying it
is a *compatibility profile*, when it is sent, and that it must not be used
anywhere else.

### 4.3 Conditions, count, and gating

- Fires when the hop's outcome classifies as `blocked` **or** `stall`.
- Fires **once per fetch**, not once per hop — a redirect chain cannot multiply
  it — and never on an enrichment probe (`_try_enrichment` keeps failing silently;
  an enrichment probe is an optimisation and must not spend a second request).
- Config-gated: `web_fetch.blocked_retry: true` by default. Default on because
  the measured win is real and the cost is one request on an already-failed
  fetch; the switch exists for an operator who wants the client to be honest even
  in the face of a refusal, and for a clean A/B during rollout.
- The escalated attempt gets the same deadline arithmetic as any other.
- If the escalated attempt also fails, the **escalated** outcome is what is
  reported (it is the more informative one: canadiantire goes from "timed out"
  to "403 from Akamai"), with `details` recording both.

---

## 5. The model-facing result for a confirmed block

### 5.1 `is_error` / `ok` stay as they are

A block remains `is_error=True`, `ok=False`, `http_error=True`. It is a failure
to retrieve the page; nothing about better diagnostics changes that, and the card
and the row model already branch correctly on it (`tool_card.py:782-793`,
`tool-row-model.ts`). **No enum or shape change** — this is what keeps the second
repo out of the blast radius (§8.2).

### 5.2 The challenge body is not inlined — decision, with the reasoning

**Make the call: do not inline the challenge body. Replace it with a one-line
origin statement plus the extracted reference id.**

The body is unusable by construction — it is markup whose entire purpose is to be
executed by a browser. Inlining it costs ~5.5 KB of context (medium.com,
measured) to tell the agent nothing, and the current presentation actively
invites the misread that this is page content. The counter-argument — that a
block page occasionally carries a human-readable reason — is real for *other*
non-2xx classes, which is why this narrowing applies **only to the `blocked`
class**. A 404's body, a 451's body and a 500's body still render exactly as
today: some of them genuinely explain themselves.

The reference id is the one durable piece of a block page (`Reference
#18.47182117.…` is what a human would quote to the site's support), so it is
extracted by regex and carried on a single line. Cloudflare's `cf-ray` is
carried the same way.

New preview for a confirmed block:

```
⚠ HTTP 403 Forbidden — blocked by Akamai bot protection, not page content.
https://www.shoppersdrugmart.ca/?lang=en&query=power+bar
markdownify · text/html · cache miss · 2 attempts (default, browser-profile)

The origin's bot protection refused this request. A browser-shaped retry was
also refused, so no headless fetch of this URL will succeed.
Origin reference: 18.47182117.1789248907.62349a41
Next step: use the `browser` tool on this URL — it drives the real browser
(a write-tier, approval-gated action), which is the only path that clears an
interactive challenge.
```

And for the stall-turned-block (canadiantire):

```
⚠ HTTP 403 Forbidden — blocked by Akamai bot protection, not page content.
https://www.canadiantire.ca/en/pdp/…-0527211p.html
markdownify · text/html · cache miss · 2 attempts (default timed out after 12.0s,
browser-profile)
…
```

And when nothing is retrievable at all (both attempts stalled), the terminal text
names the class and the count rather than trailing off:

```
read timed out after 20.0s (2 attempts: default, browser-profile) —
the origin accepted the connection but never sent a response.
https://…
```

That last line is the direct fix for the empty-`str(exc)` defect in §1.3: the
message is built from the **classified** failure, never from `str(exc)`, so it
can never be empty. `str(exc)` is appended only when it is non-empty.

### 5.3 `details` keys — additive only

Added to the existing `result_like` mapping (`tool.py:452-464`). Every key is
new; no existing key changes type, name or meaning, so `_fetch_result_output`
(`tool_card.py:744`), the UI row model, and any stored transcript keep working
untouched:

| Key | Type | Meaning |
|---|---|---|
| `attempts` | `int` | total network attempts for this fetch (1 when nothing retried) |
| `failure_kind` | `str` | one of `transport`/`stall`/`server`/`ratelimit`/`blocked`/`client`, absent on success |
| `block_vendor` | `str \| None` | `cloudflare`/`akamai`/`datadome`/`perimeterx`/`imperva`, or absent when unmatched (§2.2) |
| `block_reference` | `str \| None` | the extracted origin reference / `cf-ray` |
| `profile` | `str` | `default` or `browser` — which profile produced the reported outcome |
| `retry_after_s` | `float \| None` | the honoured or reported `Retry-After` |
| `suggested_tool` | `str \| None` | `"browser"` when the block is confirmed unretrievable |

A cache hit continues to carry none of these except `attempts: 0`, which is
truthful (no network attempt was made).

### 5.4 The card: no structural change, and why that is the right call

`_fetch_result_output` builds its rows from `details`
(`tool_card.py:744-809`). Two things happen with **no code change at all**
because of how the card already works:

1. The error row keeps rendering from `http_error` + `status`.
2. The new body text flows through `_strip_fetch_header` correctly — verified by
   running the real function against the proposed new header block in this
   worktree: the lead and meta lines are stripped and the new explanatory lines
   survive, and the existing 404 shape still strips identically.

I therefore recommend **not** adding a card hint row in this change. The reasons:

- The information the agent needs is model-facing text; the human reading the
  card already sees `⚠ HTTP 403 Forbidden — error/block page`, which is accurate.
- A new row means a TUI change, a designer round, **and** a parity obligation in
  `local-operator-ui` (`docs/evidence/tui-parity/tool-row-spec.md:282` pins the
  fetch header rows in order). That is a second repo's work for a cosmetic gain.
- The card body already prints the new explanation text under the structured
  rows, so the "next step: browser" sentence **is visible to the user** — it
  arrives as body, not as a pinned row.

Because the card's *painted output changes* (the body text differs even though no
row logic does), a rendered before/after frame is still required by the
repository's visual-validation rule — see §7. This is a "capture the frame, no
designer round" change: there is no layout, spacing, colour or row-structure
delta to review. If the reviewer disagrees and wants the hint pinned as a row,
that flips it into a user-visible change with a designer round and a parity PR;
§12 lists it as a decision.

---

## 6. Files, functions, and tests

### 6.1 New: `local_operator/web_fetch/failure.py` (~140 lines)

- `FetchFailure` frozen dataclass: `kind`, `retryable`, `vendor`, `reference`,
  `detail`, `status`, `retry_after_s`.
- `classify_exception(exc) -> FetchFailure` — the httpx exception taxonomy.
- `classify_response(status, headers, body_head) -> FetchFailure | None` — the
  status + marker taxonomy; `None` for 2xx/3xx.
- `BLOCK_MARKERS` — the header and body marker tables from §2.1, each entry
  carrying its vendor label, with a comment recording the live capture that
  justifies it (`cf-mitigated: challenge` on medium.com; `server: AkamaiGHost`
  on shoppersdrugmart.ca).
- `extract_reference(body_head, headers) -> str | None`.
- `describe(failure, attempts, profiles) -> str` — the one place the
  model-facing sentences are composed, so tool text and card text cannot drift.

A new module rather than more mass in `service.py` (already 777 lines): this is a
pure, table-driven, easily-tested classifier with no I/O, and keeping it pure is
what makes the marker tests cheap.

### 6.2 `local_operator/web_fetch/service.py`

- `USER_AGENT` (`:65`) unchanged. Add `BROWSER_PROFILE_HEADERS` beside it with the
  §4.2 set and the "only on a refusal" comment.
- Add `_RETRY_BASE_S`, `_RETRY_CAP_S`, `_RETRY_JITTER`, `_STALL_FIRST_ATTEMPT_
  FRACTION` module constants, each commented with the measurement in §3.
- `_stream_once` (`:597-637`): accept `extra_headers` and a per-request
  `httpx.Timeout`; on exception, raise a `FetchError` **carrying** the
  `FetchFailure` (add a `failure` attribute to `FetchError` — additive, existing
  `str(error)` call sites at `tool.py:377,443` keep working). Also return the
  first 8 KB of body separately for marker scanning, avoiding a second decode.
- `_follow` (`:510-551`): becomes the retry loop. Per hop, call a new
  `_attempt_with_retries(...)` that owns attempts, classification, backoff and
  the deadline check. The `validate_public_url` + `_pin_request` calls stay
  exactly where they are and run **per attempt**, not per hop — so a retry
  re-validates, and a host that becomes private between attempts is refused.
  **This is the single most important line of the change** and gets its own test.
- `_fetch_owned` (`:492-508`): computes the deadline, owns the one-per-fetch
  escalation budget, passes both down.
- `_render` (`:639-693`): for a `blocked` classification, substitute the
  one-line statement for the challenge body (§5.2) and set the new fields.
- `coerce_fetch_settings` (`:87-106`): clamp the two new numeric knobs.

### 6.3 `local_operator/web_fetch/models.py`

Additive on `WebFetchSettings` and `DEFAULT_WEB_FETCH_CONFIG` (`:68`):

```python
max_attempts: int = 3          # per hop; clamped 1..5 (1 = today's behaviour)
blocked_retry: bool = True     # one browser-shaped attempt after a refusal
```

Plus the new optional `FetchResult` fields (`attempts`, `failure_kind`,
`block_vendor`, `block_reference`, `profile`, `retry_after_s`), all defaulted so
every existing construction site stays valid.

**Migration safety — checked, and the answer is "nothing to do."**
`config_migrations.py` has no `web_fetch` handling at all; the only mention is a
comment at `:160` noting that the cleanup migration's rewrite fills in absent
top-level defaults. Settings load through `coerce_fetch_settings`, which merges
`DEFAULT_WEB_FETCH_CONFIG` under the user's mapping (`service.py:95-97`), so an
existing `config.yml` without the new keys gets the defaults. Verified
empirically that `WebFetchSettings` is `extra="ignore"`, so a config written by a
*newer* lop and read by an older one is also safe. **No migration is needed and
none should be written** — adding one would be a migration that does nothing.

### 6.4 `local_operator/web_fetch/tool.py`

- `_header_line` (`:156-181`): append ` · N attempts (…)` to the meta line when
  `attempts > 1`; the block class gets the §5.2 lead. The card's strip regexes
  (`_FETCH_HEADER_META_RE`, `tool_card.py:841`) match structurally on
  `… · … cache ` and were **verified in this worktree** to still strip the new
  meta line.
- `run_fetch` (`:452-464`): populate the new `details` keys.
- `:478` cacheability: unchanged — `http_ok` still gates it, so no retried
  failure is ever cached. Add a test, not a code change.
- `_DESCRIPTION` (`:56-68`): one added clause telling the model that a confirmed
  block names `browser` as the next step. This is the cheapest possible
  improvement to the flail behaviour.

### 6.5 `local_operator/web_fetch/cli.py`

`fetch status` gains an `attempts`/`blocked-retry` line; `fetch set` gains
`attempts` and `blocked-retry` keys. Small, mirrors the existing setters.

### 6.6 Unit tests to add (`tests/unit/web_fetch/`)

**`test_failure.py` (new)** — pure classifier:
1. `cf-mitigated: challenge` header → `blocked`, vendor `cloudflare`.
2. `Just a moment...` body with no branded header → `blocked`, `cloudflare`.
3. `server: AkamaiGHost` + `Access Denied` body → `blocked`, `akamai`, and
   `extract_reference` returns `18.47182117.1789248907.62349a41`.
4. `x-datadome` header → `blocked`, `datadome`.
5. **Unmatched 403** → `blocked`, `vendor is None`, and `describe()` does **not**
   contain the word "bot" (§2.2 — the anti-mislabelling assertion).
6. 404 → `client`, `retryable is False`.
7. 500/502/503/504 → `server`, retryable.
8. 429 with `Retry-After: 7` → `ratelimit`, `retry_after_s == 7.0`; and with an
   HTTP-date value; and with garbage → `None`.
9. `httpx.ReadTimeout` → `stall`; `ConnectError` → `transport`.
10. A 5 MB 200 body containing `access denied` at offset 3 MB → **not** blocked
    (the 8 KB scan window).

**`test_service.py` (extend)**:
11. 500 then 200 (MockTransport) → one `FetchResult`, `attempts == 2`, content
    from the 200.
12. Persistent 500 → error after exactly `max_attempts` requests; the transport
    call count is asserted.
13. 404 → exactly **one** request (non-retryable classes are not retried).
14. 429 with `Retry-After: 1` → honoured, slept once (clock monkeypatched).
15. 429 with `Retry-After: 900` → **not** slept, returns immediately,
    `retry_after_s == 900`.
16. **SSRF across retries**: a host that resolves public on attempt 1 and
    `127.0.0.1` on attempt 2 → the retry is refused, not followed. The
    resolver mock asserts `validate_public_url` was called once per attempt.
17. **Pinning across retries**: every attempt, including the browser-profile one,
    targets the vetted IP with the original `Host` and `sni_hostname`.
18. **Blocked-retry**: 403+challenge on the default profile, 200 on a request
    carrying the Chrome UA → success, `attempts == 2`, `profile == "browser"`.
19. **Blocked-retry fires once**: 403 on both → exactly 2 requests, no third.
20. `blocked_retry: false` → exactly 1 request on a 403.
21. **Deadline**: `timeout_seconds=2` against a transport that always 503s →
    total elapsed < 2.5 s and fewer than `max_attempts` requests when the
    deadline binds (asserted structurally on a fake clock, per AGENTS.md
    "prefer a structural invariant to a numeric one").
22. **Stall shortening**: a transport that never responds → attempt 1 is given
    ≤ 0.6·T, and the escalated attempt happens.
23. **Empty-`str(exc)` regression**: a `ReadTimeout("")` produces a message that
    names the class and the attempt count and does **not** end in `': '`.
24. Enrichment probes never trigger the escalation (a blocked `.md` twin costs
    exactly one request).
25. `max_attempts: 1` reproduces today's exact behaviour byte-for-byte on a 200.

**`test_tool.py` (extend)**:
26. **No-cache-on-non-2xx across retries**: 500,500,500 → nothing written to the
    cache dir; then 200 → cached. Asserted on the cache directory, not a mock.
27. A retried *success* IS cached (the 500→200 case writes an entry).
28. `details` carries `attempts`, `failure_kind`, `block_vendor`,
    `suggested_tool` on a block, and existing keys are unchanged (an explicit
    "old keys still present with old types" assertion).
29. **Abort during backoff** returns promptly (structural: the abort signal is
    set during the sleep and the result is the abort result).
30. A blocked result's preview does **not** contain `Just a moment` /
    `<html` — i.e. the challenge body is genuinely gone — but **does** contain
    the reference id and the `browser` next step.
31. A 404's body IS still inlined (the §5.2 narrowing is only for `blocked`).

**`test_efficiency.py` (extend)**:
32. The blocked preview is < 400 chars where today's medium.com preview is
    ~5.5 KB of markup — the context-cost regression guard.

Per AGENTS.md "prove the test can still fail", the retry-count and
escalation tests get a documented mutation check in the PR evidence (set
`max_attempts=1`, watch 11/12/18 fail).

---

## 7. Live evidence plan

All of it runs under `env HOME=/tmp/wf-evidence
LOCAL_OPERATOR_CONFIG_DIR=/tmp/wf-evidence/.local-operator` per the AGENTS.md
isolation rule (config dir alone is not enough), against the **worktree's own
venv**, and both a before (on `origin/main` in a throwaway checkout) and an after
capture go on the PR.

1. **medium.com: 403 → 200.**
   `.venv/bin/python -m local_operator.cli fetch test "https://medium.com/"`
   Before (captured today): `error: ⚠ HTTP 403 … / Just a moment...`.
   Expected after: `[200] https://medium.com/ … · 2 attempts (default,
   browser-profile)` and real article text in the preview.
2. **The stalled host becomes a bounded, informative result.**
   Same command against the canadiantire PDP URL.
   Before (captured today): 20.5 s, `error: timed out fetching '…': ` (empty
   reason). Expected after: ~12.3 s, `⚠ HTTP 403 Forbidden — blocked by Akamai
   bot protection` with the reference id and the `browser` next step, and the
   elapsed time printed alongside.
3. **Local server that 500s then 200s.** A `http.server.ThreadingHTTPServer`
   subprocess on a free port, `allow_private: true` for that run only; first GET
   500, second 200. Expected: one success, `attempts: 2`, server access log
   showing exactly 2 requests, elapsed ≈ 0.5 s + jitter.
4. **Persistent 5xx proves the count and the bound.** Same server, always 503.
   Expected: exactly 3 requests in the access log, total elapsed ≈ 1.9 s + 3×RTT
   and comfortably under `timeout_seconds`; error text naming
   `503 Service Unavailable (3 attempts)`.
5. **Deadline holds.** Same server with a 30 s sleep, `--timeout-seconds 3`.
   Expected: returns in < 3.5 s, not 3 × 3 s.
6. **`Retry-After` honoured and refused.** Local server returning 429 with
   `Retry-After: 1` (expect one ~1 s sleep then success) and with
   `Retry-After: 600` (expect immediate return, no sleep, the number in the text).
7. **The shopper's URL proves the new diagnostic, honestly.** The original four
   URLs from the incident. Expected: still failures — this design does not claim
   to retrieve them — but each now says Akamai, carries the reference id, and
   names `browser`. The PR states plainly that `shoppersdrugmart.ca/` and
   `/?<query>` return 403 to every client tested (curl with a full Chrome header
   set, httpx with HTTP/2 and a cookie jar, all profiles), while
   `https://www.shoppersdrugmart.ca/en` returns 200 — so the fix here is
   diagnosis and escalation, not retrieval.
8. **A normal 200 is unchanged.** `example.com`, a JSON endpoint, a markdown
   file, and a PDF fetched on `origin/main` and on the branch with the same
   isolated config; the previews are diffed and must be **byte-identical**, and
   `attempts` must be 1. This is the regression that matters most.
9. **Card frame.** Per AGENTS.md visual validation: drive the real TUI with
   `run_test`, put a blocked fetch card on screen, `app.save_screenshot()`,
   render and *look* at the SVG — before and after, consecutive frames — plus the
   geometry numbers. No designer round is requested (§5.4) but the frames are
   attached.
10. **QA independence.** The qa-tester repeats 1-8 from its own matrix on the
    branch, including the wrong-input cases (`--max-bytes 1`, a private URL, a
    `file://` URL) to confirm the SSRF and validation paths are untouched.

---

## 8. Risks to watch on rollout

1. **SSRF re-validation on the retry path.** The one way to get this badly wrong
   is to hoist `validate_public_url`/`_pin_request` out of the attempt loop "for
   efficiency", which would let a rebinding host be re-contacted on attempt 2
   without a check. Test 16/17 exist for exactly this; it is the reviewer's top
   focus.
2. **The honesty of the browser-shaped profile — stated plainly.** On a blocked
   retry we send a Chrome User-Agent and browser-shaped headers. That is a
   deliberate misrepresentation of the client, and it deserves to be named rather
   than buried. It is proportionate because: it is sent only *after* the origin
   refused an honest, self-identifying request; it is one extra request, not a
   posture; it sends no cookies and no credentials, so it does not impersonate a
   *user*, only a generic browser; the operator can turn it off
   (`blocked_retry: false`); and the alternative — telling the user "this page
   cannot be read" when a header set would have read it — is worse service for no
   ethical gain. It does **not** attempt to solve challenges, execute challenge
   JS, rotate identities, or use a proxy. Those would be circumvention; this is a
   retry.
3. **`WebReadIO` client-cache interaction.** Clients are keyed by
   `("fetch", scheme, host, port, timeout, id(transport))` (`service.py:535`).
   Moving to a per-request timeout means `timeout` stops varying that key, which
   is *good* (fewer pool entries) but must not accidentally share a client across
   two different timeouts in a way that changes behaviour — it cannot, because
   the timeout now rides on the request. Watch the pool-eviction path
   (`io.py:70-79`) under the new attempt volume; the ≤32 idle bound still holds.
4. **Singleflight and retries.** `run_fetch` collapses duplicate in-flight reads
   (`tool.py:420-435`). Retries happen *inside* the shared coroutine, so N
   subscribers still cost one retry sequence — correct, and worth a note so
   nobody "fixes" it into per-subscriber retries. A cancelled final subscriber
   cancels the retry loop mid-backoff, which the abort test covers.
5. **Politeness / extra origin load.** Worst case this triples GET volume to an
   origin that is already failing. Mitigated by: retries only on classes that
   indicate the origin *wants* another try, `Retry-After` honoured, jitter to
   avoid synchronised bursts from parallel fetches, a hard 3, and no retry at all
   on 4xx. Net new load on a healthy origin is **zero** — a 200 still costs one
   request.
6. **False-positive block classification.** A legitimate page containing "Access
   Denied" in its text could in principle be misclassified — but only on a
   non-2xx response, since `classify_response` is never consulted for a 2xx. The
   8 KB window and the header-first ordering keep this small. Test 10 guards it.
7. **Marker drift.** Vendors rename things. The failure mode is graceful: an
   unmatched 403 still gets the escalation and an honest unbranded message
   (§2.2). The markers are in one table with captured evidence in comments, so
   refreshing them is a small, obvious edit.
8. **Latency on the block path.** +0.25 s on a failing fetch. Acceptable; the
   stall path is 8 s *faster*.

### 8.1 The approval boundary — decisive, and it constrains the design

`web_fetch` is `approval_tier="read"` (`tool.py:545`); `browser` is
`approval_tier="write"` (`builtin.py:9677`, whose comment says it "navigates and
can write a screenshot file, so it rides the write approval gate rather than
auto-approved read").

**`web_fetch` must therefore never drive the paired browser itself, on any code
path, as any kind of fallback.** Doing so would launder a write-tier,
approval-gated action — navigating the operator's real, logged-in browser — through
an auto-approved read call the user never approved. It would also hand that
capability to any subagent holding `web_fetch`, silently widening the tool
surface past what the subagent was granted. The escalation stays a **named next
step the agent takes through the `browser` tool**, subject to its own approval
prompt. The design's contribution is to make that next step *legible* in the
result text rather than something the agent has to guess at.

This is also why §5.2's text says "use the `browser` tool" rather than doing it.

### 8.2 Second-repo (parity) exposure

`local-operator-ui` ports the row rules
(`docs/evidence/tui-parity/tool-row-spec.md:282` pins the fetch header rows in
order). Because §5.4 adds **no row** and §5.3 adds **only optional details keys**,
the parity spec stays true and no UI PR is needed. Verified that the UI's row
model keys off the tool name and category only
(`renderer/src/features/chat/components/trace/tool-row-model.ts:389`) and reads
none of the fetch detail keys, so new keys are invisible to it. If §12's decision
3 goes the other way, that changes and a parity PR is in scope.

---

## 9. What this design does NOT do, and why

1. **No remote reader backends.** No `r.jina.ai`, no Parallel, no third-party
   proxy — `docs/design/web_fetch.md` rejects them because they send the user's
   URL to a third party. Unchanged and reaffirmed.
2. **No headless browser, no JS execution, no challenge solving.** That is what
   the `browser` tool is, it is write-tier for good reason (§8.1), and a second
   browser stack in the fetch path would be both a footprint disaster and an
   approval-boundary breach.
3. **No cookie jar, no session persistence, no login.** `web_search/io.py:25-30`
   deliberately refuses cookies; a fetch is not the user's session. A retained
   jar was measured not to flip shoppersdrugmart anyway.
4. **No proxy support, no IP rotation, no residential exit.** Circumvention, not
   robustness.
5. **No per-site rules, allowlists, or URL rewriting.** The controls in §1.2 are
   the argument: most sites are fine with the honest UA, and a site table is a
   maintenance liability that rots silently. In particular, no "rewrite
   `shoppersdrugmart.ca/` to `/en`" special case, even though it would have
   "fixed" the incident URL.
6. **No caching of failures, and no negative cache.** `tool.py:478` stays as it
   is; a transient outage must never be replayed for the TTL.
7. **No change to the default request identity** (§4.1).
8. **No retry of the enrichment probes.** They are an optimisation; retrying them
   multiplies request volume for no user-visible gain.
9. **No new card row, no designer round** (§5.4) — unless §12.3 says otherwise.
10. **No config migration** (§6.3) — additive defaults already cover old files,
    and a no-op migration is worse than none.
11. **No change to `_fetch_or_abort`** — it already covers everything it needs to
    (§3.5).
12. **No fix for `shoppersdrugmart.ca/?query=…` itself.** It cannot be fetched
    headlessly by any client we can honestly be; the design improves the
    diagnosis and names the escalation, and the PR will say so in those words.

---

## 10. Alternatives considered and rejected

- **Just raise the timeout.** Makes the canadiantire case *worse* (longer stall,
  same nothing) and does not touch the block case.
- **Make the browser-shaped profile the default.** Flips medium.com, but pays a
  misrepresentation on every request including the ~100 % that do not need it,
  and removes site owners' ability to identify us. Rejected (§4.1).
- **Retry everything, including 4xx.** Wasted requests; a 404 will not become a
  200. Rejected.
- **A generic retry decorator around `service.fetch()`.** Simpler to write, but
  it retries the *whole* redirect chain and the enrichment probes, multiplying
  request volume, and it cannot re-validate per hop. Rejected in favour of
  per-attempt retries inside `_follow`.
- **Do nothing and tell the agent to use `browser` in the prompt.** Genuinely
  considered — it is the smallest change. Rejected because the `browser`
  extension was not paired in the incident, so the escalation was unavailable;
  because it does nothing for the transient and stall classes; and because the
  ~5.5 KB challenge body would still be burning context.

---

## 11. Relationship to `docs/design/web_fetch.md`

Nothing here contradicts it. Three places refine it:

- §7 item 7 of that doc says "Set a plain `User-Agent` identifying lop". That
  remains the default and the posture; this document adds a **single
  compatibility attempt after a refusal**, which is a narrowing exception, not a
  reversal. The original doc's intent — no ambient credentials, no user session —
  is fully preserved.
- §6 of that doc describes one request per hop; this makes it up to three plus one
  escalation, under a whole-call deadline that did not previously exist.
- §16's risk list gains the four risks in §8 above.

`docs/design/web_fetch.md` gains a one-line pointer under its title directing
readers here for the retry/blocked-response contract. **Recommendation: a
separate document (this one) plus that pointer**, rather than a §17 appended to
the existing doc — the original is a shipped feature's design record and reads as
a coherent whole; bolting a post-incident robustness contract onto its end would
bury it. A separate, cross-linked document is easier to find and easier to
supersede.

---

## 12. Decisions I want the manager to make before coding

1. **`blocked_retry` default.** I recommend **true** (measured win, one request,
   only after a refusal). Flipping it to false-by-default makes the feature
   nearly dead code, since nobody will find the switch. Confirm.
2. **The 0.6 stall fraction** (§3.3). It is the only number here that is a
   judgement rather than a measurement. I recommend shipping 0.6 and watching it;
   the evidence that would settle it is a distribution of time-to-first-byte
   across slow-but-working origins, which we would have to collect. Confirm, or
   tell me to make it configurable (I lean against: another knob nobody tunes).
3. **Card hint row: no.** I recommend the model-facing-text-only change with a
   rendered frame but **no designer round and no UI parity PR** (§5.4, §8.2). If
   you want the `browser` escalation pinned as a structured card row instead, say
   so now — it adds a designer round here and a parity PR in `local-operator-ui`.
4. **Scope of the empty-error fix.** The manager flagged it as "fold it in if
   cheap". It is cheap and it is the same code path (the terminal failure text is
   built by `describe()` either way), so I have folded it in. Confirm you want it
   in this PR rather than split.
5. **Ship-or-hold.** This is a bounded robustness fix to an existing tool with no
   new dependency, no new tool, no schema break and no security-model change. It
   touches the SSRF-adjacent retry path, which is why §8.1 and tests 16/17 exist.
   Under the team's default disposition my read is **ship**, with the reviewer
   told to scrutinise the per-attempt re-validation specifically. Confirm.
