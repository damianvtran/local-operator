# Design: `web_fetch` — a token-efficient web fetch capability for lop

Status: proposal (architect). Do not implement from this without the manager's
go-ahead on the flagged decisions at the end.

Grounded against the tree at the time of writing; file:line citations are load-
bearing, not decoration.

---

## 1. The problem, as the code actually has it

- lop has `web_search` — a load-balancing federated search
  (`local_operator/web_search/`), whose tool description literally tells the
  model to reach for the browser when it needs a page:
  `"call browser on a URL when the full page is needed"`
  (`web_search/tool.py:353`). The same instinct is baked into the search TUI
  card footer: `"Ask Operator to open result N with browser for the full page."`
  (`tui/widgets/tool_card.py:513`).
- The `browser` tool needs the user's cmux surface
  (`ToolContext.browser`, `harness/types.py:543`) and degrades to a
  single-call surface the session can never close when absent. It is the wrong
  instrument for a headless server, a subagent, or a CI/exec-mode run — exactly
  the contexts where fetching a page is most needed.
- `read` already dispatches internal URL schemes but **routes `http(s)://` to
  the filesystem** and fails: `execute_read` treats a target as an internal URL
  only when it contains `://` *and does not* start with
  `("http://", "https://", "file://")` (`tools/builtin.py:2162`). So
  `read https://example.com` falls through to `_resolve_workspace_path` and
  errors with "Path does not exist".
- Base install carries only `httpx` (`pyproject.toml` dependencies) — **no**
  HTML-rendering library at all (markdownify / html2text / trafilatura /
  readability / bs4 / lxml / selectolax all absent — confirmed).

Net: an agent that finds a URL via `web_search` has no first-class,
headless-safe way to read it. That is the gap.

---

## 2. The central decision — dedicated tool vs. overloading `read`

### Recommendation: **a dedicated `web_fetch` tool, and ALSO teach `read` to
accept `http(s)://` as sugar that delegates to the same engine.** The hybrid.

This is the one place I depart from "smallest possible change", and I want to
justify why the hybrid earns its keep here rather than being over-built.

**Why a dedicated tool at all** (not read-overload alone):

- **Approval tier and semantics differ from a file read.** `read` is tier
  `"read"` and is auto-approved by the host; it is synchronous, local, and
  can't touch the network. A network fetch has SSRF surface, redirects,
  timeouts, and a `max_bytes` knob that a file read has no concept of. Folding
  all of that into `read`'s param schema and description muddies the one tool
  every agent uses hundreds of times a session. `AgentTool` already carries
  `approval_tier` / `concurrency` / `interruptible` per tool
  (`harness/types.py:640-642`); a fetch wants `interruptible=True` and its own
  clean description, which a dedicated tool gives for free and an overload
  cannot.
- **A clean TUI card.** `tool_card.py` already special-cases `web_search`
  (`:1163`, `:1560`, `:1851`) to render provider/sources rows from structured
  `details`. A `web_fetch` card wants its own shape (final URL, status,
  content-type, render method, byte/line counts, spill handle) — trivial as its
  own `tool_name` branch, awkward if every `read` of a file has to first ask
  "was this a URL?".
- **Discoverability.** A tool named `web_fetch` sitting next to `web_search` in
  the tool list is the obvious verb after a search. The pairing *is* the mental
  model: search → fetch.

**Why also overload `read`** (the sugar):

- The user's stated tie-breaker is "whichever is easier and more intuitive for
  the agents to reach for." Agents already `read spill://…`, `read skill://…`,
  `read <file> range=…`. Making `read https://…` *work* (instead of failing to
  the filesystem as it does today at `builtin.py:2162`) removes a sharp edge and
  costs one branch. It is not a second convention — it is the SAME convention
  spill already established: spill.py's own docstring frames a handle as "just
  another internal URL that `read` resolves" (`spill.py:14-17`). A URL is the
  most natural internal-ish URL there is.
- Crucially, **the sugar and the tool share one engine and one output shape.**
  `read https://…` calls the fetch engine, gets back the same bounded preview +
  `spill://` handle, and returns it. There is no duplicated pipeline — just a
  second doorway into it. Chunked re-reads then go through `read spill://…`
  exactly as they do for every other spilled output.

**What I explicitly reject:** overloading `read` *instead of* a dedicated tool.
It saves one tool definition but pays for it with a bloated `read` description,
a mixed approval story, and a fetch that has to smuggle its params (raw,
max_bytes, timeout) through `read`'s schema — where a `range` on a URL would
have to mean "fetch then slice", conflating a network op with a slice op. The
dedicated tool is where the params and the card live; the `read` sugar is a
thin, forwarding convenience with **no** extra params (a URL passed to `read`
uses fetch defaults; anyone who wants `raw`/`max_bytes` uses `web_fetch`).

> Note: omp — the reference — chose pure read-overload (`executeReadUrl`,
> `parseReadUrlTarget` in `fetch.ts`), because omp's `read` already carries URL
> line-selector syntax (`:50-100`, `:raw`). lop's `read` has a cleaner param
> model (`range`, `raw` fields, `ReadParams` at `builtin.py:1519`) and a
> separate-tool culture (`web_search` is its own module). The hybrid fits lop's
> grain better than cloning omp's single-verb choice.

---

## 3. Tool shape

**Name:** `web_fetch`

**Description string (one paragraph, model-facing):**

> "Fetch a web page or file over HTTP(S) once and make it readable without
> pulling the whole thing into context. Returns a short preview plus a
> `spill://` handle; expand it with `read` using a line range or
> `?q=<regex>` search, exactly like any other large output. Renders HTML to
> clean markdown, pretty-prints JSON, returns text as-is, and reports type and
> size for PDFs/binaries instead of inlining them. Re-fetching the same URL
> within the cache TTL reuses stored content with no network call. Use this
> (or `read <url>`) instead of `browser` for headless, subagent, and
> server contexts; use `browser` only when a page needs a real logged-in
> session or JavaScript rendering."

**Params (`WebFetchParams`, `extra="forbid"`, mirroring `WebSearchParams` at
`web_search/tool.py:42`):**

```python
class WebFetchParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str = Field(description="Absolute http:// or https:// URL to fetch.", min_length=1)
    raw: bool = Field(
        default=False,
        description="Return the source verbatim (no HTML→markdown rendering, no JSON pretty-print).",
    )
    max_bytes: int | None = Field(
        default=None, ge=1,
        description="Override the configured download ceiling for this call.",
    )
    timeout_seconds: float | None = Field(
        default=None, gt=0,
        description="Override the configured request timeout for this call.",
    )
    refresh: bool = Field(
        default=False,
        description="Bypass the cache and force a fresh network fetch.",
    )
```

**AgentTool wiring** (`build_web_fetch_tool`, mirroring
`build_web_search_tool` at `web_search/tool.py:343`):

- `approval_tier="read"` — a fetch reads a remote resource and produces no
  local side effect beyond the bounded spill/cache under `config_dir()`; it
  belongs in the same auto-approvable tier as `web_search` (which is also
  `"read"`, `web_search/tool.py:357`). SSRF is handled by the URL policy in §7,
  not by forcing an approval prompt on every fetch. (See the flagged decision
  in §12 on whether outside-workspace-style escalation should apply.)
- `concurrency="shared"`, `interruptible=True` — same as web_search; a fetch is
  network-bound and must be abortable for steering.
- `execute=execute_web_fetch`.
- createIf gate: return `None` when `values.web_fetch.enabled` is false,
  reading the startup snapshot the same way web_search does (see §9).

---

## 4. Module / file layout

Mirror `web_search/` exactly — this is the established pattern and matching it
is the point.

```
local_operator/web_fetch/
  __init__.py
  models.py     # WebFetchSettings, FetchResult, DEFAULT_WEB_FETCH_CONFIG, SSRF policy enum
  render.py     # HTML→markdown / JSON / text / binary classification (the pipeline core)
  service.py    # settings load/save, cache index, SSRF validation, the fetch+render+cache orchestration
  tool.py       # WebFetchParams, execute_web_fetch, build_web_fetch_tool, model-facing render
  cli.py        # `lop fetch` status / test / config subcommands (see §10)
```

Touch points in existing files (all additive):

- `tools/registry.py`: add `"web_fetch": lambda context: build_web_fetch_tool(context)`
  to the factory table (`registry.py:37`) and to the stable order list
  (`registry.py:63`).
- `tools/builtin.py`: in `execute_read`, before the internal-URL branch at
  `:2162`, add an `http(s)://` branch that delegates to the fetch engine (the
  `read` sugar). One new branch, ~6 lines.
- `harness/types.py`: add `web_fetch_settings: dict[str, Any] | None = None`
  to `ToolContext` (beside `web_search_settings` at `:526`).
- `session_factory.py:1013` and `harness/subagent.py:696`: populate
  `web_fetch_settings` from config, exactly as web_search is populated.
- `config.py:173`: add `"web_fetch": dict(DEFAULT_WEB_FETCH_CONFIG)` to the
  default config block.
- `cli.py:478`: register `add_fetch_subparser`.
- `tui/glyphs.py:72`: add a `"web_fetch"` glyph (globe/download variant) with
  an ASCII fallback (`:98`).
- `tui/widgets/tool_card.py`: add a `web_fetch` branch for the result card (§11).
- `web_search/tool.py:353` and `tui/widgets/tool_card.py:513`: retarget the
  "call browser" advice to `web_fetch`/`read the URL` (§13, secondary).

---

## 5. The dependency decision — base vs extra vs stdlib

### Recommendation: **base = stdlib-only renderer (always works); the good
renderer is a NEW `fetch` extra, pulled into `all`.**

Rationale, grounded in the packaging constraints the repo documents itself:

- The desktop UI installer runs a **bare** `pip install local-operator` then
  `local-operator serve` — pyproject's dependency comments call this out twice
  (`pyproject.toml` base deps comment: "The desktop UI's installer runs a BARE
  `pip install local-operator`… the default set must stay fully functional")
  and `optional.py:3-8`. So `web_fetch` **must** be useful on a bare install.
  That forbids making a rendering library a hard requirement of the tool.
- But a stdlib-only `html.parser` tag-strip produces mediocre markdown. Making
  that the *only* option undersells the feature. So: two tiers.

**Base tier (no extra, always present):** a pure-stdlib renderer in
`render.py` built on `html.parser.HTMLParser` (stdlib) that:
- strips `<script>/<style>/<nav>/<header>/<footer>` subtrees,
- keeps heading/paragraph/list/anchor/`<pre>` structure as crude markdown,
- collapses whitespace.
This is the degraded-but-useful mode required by point 4 of the brief and by
`optional.py`'s "sites that can degrade keep their own try/except" convention
(`optional.py:77-79`). It is *not* great, but it beats raw HTML and beats
failing.

**`fetch` extra (recommended default for anyone who installs `[all]`):**
`markdownify` + a lightweight sanitizer. I recommend **markdownify** over the
alternatives:

| lib | verdict |
|---|---|
| **markdownify** | ✅ Pure Python, small, depends on `beautifulsoup4` (also pure Python) + `soupsieve`. Good markdown, actively maintained, no compiled wheels → no Windows sdist-build risk (the exact fragility `optional.py:6-8` worries about). **Recommend.** |
| html2text | Pure Python, fine, but markdownify's output is cleaner for LLM consumption and bs4 gives us robust parsing we can reuse for feed/link extraction. |
| trafilatura | Best extraction quality, but heavy (pulls lxml → compiled wheels, the platform-fragility we avoid) and overkill. Support it *optionally as a subprocess backend if present on PATH* (omp does this, `fetch.ts:634`), never as a Python dep. |
| readability-lxml / lxml / selectolax | All pull compiled wheels. Reject as deps for the same Windows-fragility reason base deps are kept lean. |

So: `EXTRAS["fetch"] = "clean HTML→markdown rendering for web_fetch (markdownify)"`
in `optional.py:28`, and `fetch = ["markdownify"]` in pyproject, added to the
`all` alias (`pyproject.toml` `all = [...]`). The `require_extra` /
`missing_extra_error` machinery (`optional.py:47,73`) is exactly what the
render backend selection uses to decide markdownify-vs-stdlib and to phrase the
"install `[fetch]` for better rendering" hint — except that here we **degrade
silently to the stdlib renderer** rather than erroring, because a usable result
beats an error (the `try/except ImportError` + `missing_extra_error` warning
pattern, `optional.py:77-79`).

**Net dependency change:** one new pure-Python extra (`markdownify` +
transitive `beautifulsoup4`, `soupsieve`). Zero new base deps. Zero compiled
wheels. Bare install still works via the stdlib renderer.

---

## 6. The fetch → render → spill → cache pipeline, step by step

`execute_web_fetch` (async, in `tool.py`) → `service.fetch(url, opts)`:

1. **Validate & normalize URL.** Reject non-http(s) with the same shape as
   `_validate_browser_url` (`builtin.py:4447`) — but stronger (see §7 SSRF).
   Normalize a collapsed scheme / missing scheme cheaply (omp's
   `normalizeUrl`/`repairCollapsedScheme`, `fetch.ts:128-142`) so
   `read example.com` and a path-normalized `https:/host/x` still work.
2. **Cache lookup** (unless `refresh=True`): hash the normalized URL, look up
   the cache index (§8). On a hit whose `fetched_at` is within TTL **and** whose
   spilled digest still `stat`s in the spill store — return the stored preview +
   handle with `details.cache="hit"`, **no network call.** This satisfies
   requirement #3.
3. **Enrichment attempts (cheap, bounded, HTML pages only)** — a
   *proportionate* subset of omp's chain, in order, each with a short sub-budget:
   - `.md` suffix trick (`fetch.ts:307`): try `<url>.md` / `<path>/index.html.md`.
   - `llms.txt` candidates (`fetch.ts:99,348`): `/.well-known/llms.txt`,
     `/llms.txt`, walking up path scopes.
   - content negotiation (`fetch.ts:374`): `Accept: text/markdown,
     text/plain;q=0.9, text/html;q=0.8`.
   Each returns early if it yields substantial non-HTML content (>100 chars,
   not HTML). **I recommend shipping these three** — they are cheap, high-value,
   and increasingly common (docs sites, GitHub, many dev tools). **I recommend
   NOT shipping** omp's 80 site-specific scrapers, the Jina/Parallel remote
   reader backends (they need API keys / send the user's URL to a third party —
   a privacy call the user hasn't asked for), or the feed/notebook/sqlite/archive
   special handlers in v1 (see §12 for what to defer).
4. **Primary fetch:** `httpx.AsyncClient` (already a base dep), streaming, with
   `follow_redirects=True` bounded to N redirects (§7), the configured timeout,
   and a hard `max_bytes` read cap enforced *during* streaming (stop reading and
   mark truncated once exceeded — never buffer an unbounded body). Abort on the
   `AbortSignal` for interruptibility.
5. **Classify by content** (Content-Type + a small body sniff, never trust
   extension alone — matches `read`'s content-classification ethos,
   `builtin.py:2204`):
   - **HTML** → render (step 6).
   - **JSON** (`application/json`, or body parses) → pretty-print with
     `json.dumps(…, indent=2)` (omp `formatJson`, `fetch.ts:747`).
   - **Markdown / plain text** → pass through, whitespace-normalized.
   - **Image** → for v1, **do not inline**; report type + dimensions if cheap +
     byte size, and tell the agent to `read` the URL is not applicable — direct
     them to `browser` for visual inspection. (Inlining images to the model is a
     larger decision; defer, §12.)
   - **PDF / other binary** → **do not inline.** Return a one-line notice:
     type, byte size, final URL (omp `buildBinaryNotice`, `fetch.ts:836`). This
     is the "sane story for binary" the brief asks for.
6. **HTML render** (`render.py`): try markdownify if the `fetch` extra is
   present, else the stdlib renderer. Apply omp's **low-quality gate**
   (`isLowQualityOutput`, `fetch.ts:719`): if the output is <100 chars, or
   mostly-navigation (short-line ratio > 0.7), or JS-gated ("enable
   javascript"), keep it but flag `low_quality=true` in details so the card and
   the agent know to consider `browser`.
7. **Build the model-facing result:** a bounded PREVIEW (first ~N lines / the
   `TOOL_OUTPUT_LIMIT_CHARS` budget, `builtin.py:121`) — never the whole page.
8. **Spill the full rendered content** via the EXISTING `spill_truncate`
   (`builtin.py:437`): it writes to the spill store, returns
   `(display_text_with_footer, {"spill": {...}})`. This is the crux of the
   token-efficiency requirement and requirement #2 — the full page lands behind
   a `spill://` handle the agent expands through `read` with ranges and `?q=`,
   using the identical affordance every other oversized tool output already
   uses. **No new expansion path, no new URL convention.** If the content fits
   the budget, no spill happens and the whole thing is returned inline (also via
   `spill_truncate`, which no-ops under the limit, `builtin.py:451`).
9. **Record the cache entry** (§8): map `url_hash → {spill_digest, fetched_at,
   final_url, status, content_type, render_method, complete}`.
10. **Return** `ToolResult` with the preview text (carrying the spill footer)
    and `details` = `{url, final_url, status, content_type, render_method,
    bytes, lines, cache, low_quality, **spill_detail}` — `details` never reaches
    the provider (`builtin.py:447-449`), so it costs no tokens but drives the
    card and transcript.

The `read https://…` sugar (step in `execute_read`) calls the same
`service.fetch` and returns the same result, with the tool_name recorded as
`read` for the card.

---

## 7. SSRF / safety policy

lop's `_validate_browser_url` gates only on scheme (`builtin.py:4458`). A fetch
from arbitrary agent-chosen URLs is a real SSRF surface (cloud metadata
endpoints, internal services, `file://`, localhost admin panels). Policy:

1. **Scheme allowlist:** only `http://`, `https://`. Reject `file://`, `ftp://`,
   `gopher://`, `data:`, etc. — reuse the refusal shape of
   `_validate_browser_url` (`builtin.py:4458-4462`).
2. **Resolve-then-check, on every hop.** Resolve the hostname to IPs and refuse
   if any resolved address is private / loopback / link-local / multicast /
   reserved (`ipaddress` stdlib: `ip.is_private | is_loopback | is_link_local |
   is_reserved | is_multicast`, plus the IPv4-mapped-IPv6 and
   `169.254.169.254` metadata cases explicitly). This must run **after each
   redirect**, not just on the initial URL, or a public URL that 302s to
   `http://169.254.169.254/…` walks right through — the classic SSRF-via-redirect.
   Implement by disabling httpx's automatic redirect following and doing the
   redirect loop manually (bounded, re-validating each `Location`), OR by using
   an httpx transport/event hook that re-validates. Manual loop is clearer and
   testable; recommend it.
3. **Config-gated localhost allowance:** default **deny** private/loopback, with
   a `values.web_fetch.allow_private = false` switch. Local dev that wants to
   fetch `http://localhost:3000` flips it on knowingly. Default-deny is the safe
   posture; the switch keeps the dev path from being a papered-over exception.
4. **Bounded redirects:** `max_redirects` (default 5). Exceeded → error.
5. **Bounded bytes:** `max_bytes` (default 5 MiB — comfortably under the spill
   per-entry cap of 4 MiB rendered, and note rendered markdown is smaller than
   source HTML; the cap is on the *download*). Enforced during streaming.
6. **Bounded time:** `timeout_seconds` (default 20.0, matching web_search's
   `timeout_seconds` default, `models.py:75`).
7. **No credential leakage:** do not forward ambient cookies/auth; each fetch is
   anonymous (this is the deliberate difference from `browser`, which uses the
   user's logged-in session). Set a plain `User-Agent` identifying lop.

This is a policy the reviewer must scrutinize; it is the security-sensitive part
of the change. It is *not* a redesign of a subsystem, so it doesn't trip the
"sweeping overhaul" hold — but it is the risk to watch on rollout (§14).

---

## 8. Caching & TTL — where it lives relative to spill

**The key tension:** spill is **content-addressed** (digest = sha256 of
content, `spill.py:353`) and has **no URL→digest map**. It deliberately avoids a
global index file (`spill.py:53-59`). So spill alone cannot answer "have I
fetched this URL recently?" — fetching the same URL twice is idempotent at the
storage layer (same content → same digest → one entry) but still pays the
network round trip, which requirement #3 forbids.

**Recommendation: a thin, metadata-only cache index that points *into* spill —
NOT a second content store.**

- The **content** lives entirely in the spill store, under its existing 64 MiB
  ceiling and LRU eviction (`spill.py:113`, `_evict`). We add **zero** new
  content bytes anywhere else. This directly honours requirement #3's "do not
  invent a second unbounded store" and the 6.8 GB omp incident the spill
  docstring recounts (`spill.py:19-31`).
- The **index** is a tiny per-URL sidecar (mirroring spill's per-entry-sidecar
  choice over a global index, `spill.py:55-59`), under
  `config_dir()/web_fetch_cache/<url_hash>.json`, each ~250 bytes:
  `{url, final_url, spill_digest, fetched_at_ms, status, content_type,
  render_method, complete}`. Bounded by **count** (e.g. keep newest 500, prune
  oldest on write) — sidecars are metadata only, so even 500 is ~125 KB.

**Cache read path** (`service.fetch`, step 2 above):
1. Hash normalized URL → sidecar path. Missing → miss.
2. Load sidecar. `now - fetched_at_ms > ttl_ms` → **stale**, miss (optionally
   delete the sidecar).
3. `store.stat(spill://<spill_digest>)` (`spill.py:405`) → **None** means the
   spill entry was evicted under its own LRU; treat as miss. This is the
   critical coupling: **the cache never claims content the spill store no longer
   holds.** The index is advisory over spill's authoritative, self-bounding
   storage.
4. Otherwise → **hit**: rebuild the preview from `store.read_lines` and return
   with `details.cache="hit"`, no network.

**TTL default:** `cache_ttl_seconds = 900` (15 min) — long enough that an agent
re-reading a page it just fetched during a multi-step task pays nothing, short
enough that "fetch the latest" isn't served stale surprisingly. `refresh=True`
and a config `cache_ttl_seconds = 0` both bypass.

**Why not extend spill to key by URL?** Considered and rejected: spill's
content-addressing is a deliberate invariant (identical output = one entry,
`spill.py:44-46`), and bolting a URL keyspace onto it would complicate a module
whose whole virtue is that it is small and bounded. A separate metadata index
that *references* spill keeps each module doing one thing. The index inherits
spill's bound for free because it holds no content and self-checks against
`stat`.

---

## 9. Config schema

`values.web_fetch`, validated by a `WebFetchSettings` pydantic model in
`web_fetch/models.py`, mirroring `WebSearchSettings` (`web_search/models.py:69`)
and its `coerce_search_settings` lenient-validation pattern
(`service.py:28`). Seeded into `config.py:173` via `DEFAULT_WEB_FETCH_CONFIG`:

```python
class WebFetchSettings(BaseModel):
    enabled: bool = True
    timeout_seconds: float = 20.0
    max_bytes: int = 5 * 1024 * 1024        # download ceiling
    max_redirects: int = 5
    cache_ttl_seconds: int = 900             # 0 disables caching
    allow_private: bool = False              # SSRF: allow loopback/private targets
    render_backend: Literal["auto", "stdlib"] = "auto"  # auto = markdownify if [fetch] present
    enrich: bool = True                      # try .md / llms.txt / content-negotiation

DEFAULT_WEB_FETCH_CONFIG: dict[str, object] = {
    "enabled": True,
    "timeout_seconds": 20.0,
    "max_bytes": 5 * 1024 * 1024,
    "max_redirects": 5,
    "cache_ttl_seconds": 900,
    "allow_private": False,
    "render_backend": "auto",
    "enrich": True,
}
```

Loose-YAML coercion (`coerce_fetch_settings`) preserves safe defaults for
malformed fields, exactly as `coerce_search_settings` does
(`service.py:28-49`).

---

## 10. CLI surface

Mirror `add_search_subparser` (`web_search/cli.py:26`) with a `fetch`
subcommand — warranted because web_search set the precedent and a status view
helps users see whether the `[fetch]` extra is active and what the SSRF/TTL
policy is:

- `lop fetch status` → a `format_fetch_status` table (`cli.py` pattern at
  `web_search/cli.py:79`): enabled, render backend in effect (markdownify vs
  stdlib fallback — computed by attempting the import), TTL, max_bytes,
  max_redirects, allow_private, cache entry count / bytes.
- `lop fetch test <url>` → one live fetch, printing render method, status,
  bytes, lines, spill handle (like `_test_search`, `web_search/cli.py:178`).
- `lop fetch set <key> <value>` → focused config setters (enabled, ttl,
  allow-private, backend), each persisting only its field
  (`save_search_settings` pattern, `service.py:57`).

---

## 11. TUI card

Add a `web_fetch` branch to `tool_card.py`, following the `web_search`
precedent (`:1163`, `:1560`, `:1851`). A `_fetch_result_output(details)` helper
(sibling to `_search_result_output`, `tool_card.py:487`) renders structured rows
from `details`:

```
Fetched: https://example.com/docs   (final: https://example.com/docs/  ·  200 · text/html)
Rendered: markdownify · 342 lines · 18 KB · cache miss
<preview lines…>
Expand: read spill://<digest> (range / ?q=)   [when spilled]
```

- Use the spill footer the engine already appended for the expansion hint — the
  card shows the same `spill://` handle so a user can eyeball it.
- `low_quality=true` → a one-line amber note "sparse/JS-gated — try `browser`
  for the full page."
- Glyph: add `"web_fetch"` to `tui/glyphs.py:72` (a download/globe nerd-font
  glyph) with an ASCII fallback at `:98`.

This is a **user-visible** change (a new card), so per the team's gate it needs
a **designer** round with rendered before/after SVG frames (the card in
miss / hit / low-quality / binary-notice states) per AGENTS.md "Visual
validation" (§95-209). It does **not** change an interaction flow (no new
keybinding/widget behaviour), so a ux-reviewer round is **not** required — the
agent simply calls a tool and reads a card, which is the existing tool-call
interaction.

---

## 12. Scope: what ships in v1, what defers

**Ships (v1):**
- `web_fetch` tool + `read <url>` sugar, one shared engine.
- httpx streaming fetch, SSRF policy (§7), redirect re-validation.
- HTML→markdown (markdownify extra, stdlib fallback), JSON pretty-print, plain
  text pass-through, binary/PDF notice (type+size, no inline).
- Enrichment: `.md` suffix, `llms.txt`, content negotiation (the three cheap
  wins).
- Low-quality detection gate.
- spill integration for chunked reads; metadata-only URL cache with TTL.
- Config, CLI status/test, TUI card.

**Defers (explicitly out of v1 — name them so nobody thinks they were missed):**
- Remote reader backends (Jina `r.jina.ai`, Parallel) — they send the user's
  URL to a third party and/or need API keys; a privacy/config decision the user
  hasn't asked for.
- trafilatura/lynx **subprocess** backends — nice-to-have quality bump; can be
  added later as "use if on PATH" without touching deps.
- Feed (RSS/Atom) parsing, Jupyter notebook, sqlite, archive listing special
  handlers (omp `fetch.ts:518,879,885,899`) — long tail, low frequency.
- Image inlining to the model (returning an image block from a fetched image
  URL) — `read` already has the image-encoding machinery
  (`_encode_image_for_model`, `builtin.py:2223`); wiring it to fetched bytes is
  a clean follow-up but a separate decision about token cost.
- The 80 site-specific scrapers — deliberately never cloning these.

---

## 13. Secondary: web_search tweaks (keep small)

Once `web_fetch` exists, two low-risk, genuinely-worthwhile edits:

1. **Retarget the "call browser" advice.** `web_search/tool.py:353` currently
   ends its description with "call browser on a URL when the full page is
   needed." Change to "…use `web_fetch` (or `read <url>`) to read a result's
   full page." This is the single highest-value nudge — it points the model at
   the right, headless-safe verb. Same for the source footer constant that
   `_render_response` appends (the `_SOURCE_FOOTER` near
   `web_search/tool.py:127`) and the TUI card footer
   `tool_card.py:513` ("Ask Operator to open result N with browser…" →
   "…web_fetch result N…").
2. **Nothing else.** No provider or caching changes to web_search — that is
   scope creep and web_search's load-balancing is already good. Note it and
   move on.

These ride in the same PR only because they are one-line description edits that
keep the two tools coherent; if they complicate review, split them out.

---

## 14. Test cases that prove it works

Real execution, not just green units (per the team's testing-evidence bar).
Unit tests in `tests/unit/web_fetch/` mirroring `tests/unit/web_search/`, plus
a live smoke via `lop fetch test`.

Unit / integration (httpx `MockTransport` for determinism where possible, a
couple of real-network cases guarded like web_search's `_test_search`):

1. **Live HTML → markdown + spill + chunk-read.** Fetch a large HTML page;
   assert preview is bounded, a `spill://` handle is returned, and
   `read spill://<d> range=…` and `?q=…` both resolve the full content. This is
   the central requirement — prove the chunked path end to end.
2. **JSON pretty-print.** `application/json` body → indented output; malformed
   JSON falls back to raw text (omp `formatJson` behaviour, `fetch.ts:747`).
3. **Redirect followed + re-validated.** A 302 to a public URL succeeds and
   `details.final_url` reflects the destination.
4. **SSRF blocked — direct.** `http://169.254.169.254/…`,
   `http://localhost/…`, `http://10.0.0.1/…`, `file:///etc/passwd` all refused
   with a clear message; `http://[::1]/` and an IPv4-mapped IPv6 too.
5. **SSRF blocked — via redirect.** A public URL that 302s to
   `http://169.254.169.254/` is refused at the hop, not followed. (The case a
   scheme-only check misses.)
6. **allow_private toggle.** With `allow_private=true`, `http://localhost:PORT`
   against a local test server succeeds; with it false, refused.
7. **Large page spills, and the stored copy is head+tail when over the
   per-entry cap** (`complete=False` surfaced, `spill.py:124`).
8. **Cache hit within TTL — no network.** Fetch twice; assert the second call
   makes zero HTTP requests (MockTransport call count) and returns
   `details.cache="hit"`. Advance a fake clock past TTL → miss → network again.
9. **Cache miss when spill evicted.** Manually evict/prune the spill entry
   (`prune_all`, `spill.py:615`) between fetches; assert the cache degrades to a
   network fetch rather than returning a dead handle.
10. **Timeout.** A slow endpoint (or a MockTransport that sleeps) → clean
    timeout error naming the URL, no partial/garbage result.
11. **404 / non-2xx.** Returns the status in `details.status` and a useful
    message; does not spill an error page as if it were content (or spills it
    but flags the status — recommend: return the body but lead the result with
    the non-2xx status so the agent isn't misled).
12. **max_bytes enforced during streaming.** A body larger than `max_bytes`
    stops downloading and is flagged truncated; memory does not balloon.
13. **Binary/PDF notice.** `application/pdf` → type+size notice, no inline, no
    spill of binary bytes.
14. **`read <url>` sugar.** `read https://…` returns the same result shape as
    `web_fetch`; `read <file>` and `read spill://` are unaffected (regression
    guard on `execute_read`'s new branch).
15. **stdlib fallback renderer.** With markdownify import forced to fail, HTML
    still renders to usable text and `details.render_method="stdlib"`; bare
    install stays functional.
16. **Enrichment.** A URL whose `.md` suffix / `llms.txt` yields clean markdown
    is preferred over scraping the HTML; a site without them falls through to
    HTML render.

Plus the standard gates (flake8, black==26.1.0, isort, pyright, unit suite),
and — because the TUI card is user-visible — **rendered before/after SVG frames**
of the fetch card states per AGENTS.md §95-209, reviewed by a `designer`.

---

## 15. Decisions I want the manager to make before coding

1. **Confirm the hybrid** (dedicated `web_fetch` + `read <url>` sugar) over
   pure read-overload. I recommend the hybrid; it is slightly more than the
   minimum. If you want the truly-minimal version, drop the `read` sugar and
   ship `web_fetch` alone — the sugar is the one discretionary piece.
2. **Approval tier.** I recommend `"read"` (auto-approved), with SSRF policy as
   the guardrail. Alternative: escalate to an approval prompt for the *first*
   fetch of a session or for non-allowlisted hosts. I think that is friction
   without much gain given the SSRF policy, but it's a security posture call
   that's yours.
3. **`allow_private` default.** I recommend default **false** (deny loopback/
   private). If this machine's common workflow is fetching local dev servers,
   you may want it true — but I'd rather leave it false and flip per-project.
4. **The `fetch` extra name and its inclusion in `all`.** I recommend a new
   `fetch = ["markdownify"]` extra, added to `all`. Confirm you're happy adding
   `beautifulsoup4`/`soupsieve` (pure Python, no compiled wheels) to the `all`
   footprint.
5. **Ship-or-hold.** This is an ordinary user-facing feature (new tool + card),
   not a sweeping overhaul or a data migration — so under the team's default
   disposition it **ships** (implement → PR → full review+design gate → merge →
   `lop-update`) unless you say hold. The one thing that gives me minor pause is
   the SSRF surface; I don't think it rises to "security-sensitive redesign"
   (it's a new bounded capability with a default-deny policy, not a change to an
   existing security model), so my read is **ship**, with the reviewer told to
   scrutinize §7 specifically. Confirm.

---

## 16. Risks to watch on rollout

- **SSRF completeness** (§7) — the redirect-revalidation case (test 5) is the
  one most likely to be got subtly wrong. This is the top review focus.
- **Windows install of the `fetch` extra** — markdownify/bs4/soupsieve are pure
  Python, so risk is low, but verify the bare-install stdlib fallback (test 15)
  actually triggers when the extra is absent, since the desktop UI ships bare.
- **Spill/cache coupling** — the cache must never outlive its spill entry
  (test 9). If `stat` is skipped on a cache hit, agents get dead handles.
- **max_bytes during streaming** (test 12) — a naive `resp.text` buffers the
  whole body first and defeats the cap; the read must be incremental.
- **Not re-introducing unbounded disk** — the cache index is metadata-only and
  count-bounded; confirm no code path writes fetched *content* anywhere except
  the spill store.
