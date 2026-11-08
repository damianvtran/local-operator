<!-- Moved verbatim from the project README (2026-11-08 README refresh); the README now links here. -->

# What the Agent Can Do

The agent's built-in tools, each with its own card in the transcript:

- **Run things**: `bash` (shell commands), `eval` (a persistent Python
  kernel: variables survive across calls).
- **Work with files**: `read`, `write`, `edit` (surgical search/replace),
  `glob`, `grep`, plus `lsp` for Jedi-backed Python code intelligence.
- **Reach the web**: load-balanced `web_search` across eight providers,
  `web_fetch` for reading pages headlessly, and a `browser` tool for pages
  that need rendering, a login, or interaction.
- **Stay organized**: a visible `todo` list for multi-step work, `ask` to
  put real decisions back to you as a picker instead of a wall of text.
- **Handle secrets**: the `secret` tool stores, lists, updates and removes
  entries in the encrypted long-term store, and hands a value to `bash` or
  `eval` without ever returning it to the model. See
  [Credentials and secrets](configuration.md#credentials-and-secrets).
- **Run an organization**: `task` spawns subagents, `hub` talks to them,
  `jobs`/`wait` manage background work, `send` reaches other sessions,
  `agent` and `team` author profiles and rosters, and `wake` schedules future
  follow-ups.

## Web search

Search works out of the box. The chain walks **free legs first and metered legs
last**: the free providers you list, then the credential-free providers this
install can reach, then the best-effort ones, then anything that would spend money
or a model turn — including a paid provider you listed, which is tried first among
the paid legs and never before a free one.

```bash
lop search list
lop search test "Python 3.13 release notes"
lop search enable perplexity          # clears an exclusion
lop search disable brave             # excludes it from every search
lop search order duckduckgo tavily   # the priority prefix: free legs first, paid ids run after them
lop search setup brave --api-key
lop search setup tavily --oauth      # official Tavily MCP server
lop search setup searxng --endpoint https://search.example.com
```

`web_search.providers` is a **priority prefix, not an allowlist**, and its
authority is over order **within a band**: a named free provider joins the free
pool where you put it — and with the default `round_robin` that pool is rotated, so
"where you put it" is its position in the pool rather than a promise it is tried
first on every call (`search balance ordered` is the strict version). Every other
usable provider joins automatically behind the ones you named.
`web_search.excluded_providers` is the only way to say never — an id there is
skipped in the prefix *and* in the automatic bands. A provider that becomes
usable mid-session (a DeepSeek login, an Exa key) joins its band on the next
search with no further configuration; `lop search list` states each provider's
state, so the chain is never a guess.

Listing a provider whose transport would **spend** — a key, or a model turn —
does not put it first. It moves it to the head of the **paid** band, so every
free leg is still tried before it: a paid provider listed ahead of a free one is
tried first *among the paid legs*, and `lop search list` names it as `paid` rather
than as a plain `enabled`. That keeps the free-before-paid rule absolute, and
`search list`'s `chain:` line shows the resulting order.

`round_robin` spreads the first attempt across the whole **free pool** — the
listed free providers in their listed order, then the credential-free providers
that joined automatically — and never across a band boundary, so nothing that
spends can be rotated (or listed) ahead of a free leg.

| Provider | Access | Default |
| --- | --- | --- |
| DuckDuckGo | Credential-free | Priority prefix |
| Tavily | Keyless, `TAVILY_API_KEY`, or OAuth MCP | Priority prefix (keyed: paid band) |
| Exa | Keyless MCP (no key needed), or `EXA_API_KEY` | Automatic (free; keyed: paid band) |
| Parallel | Keyless MCP (no key needed), or `PARALLEL_API_KEY` | Automatic (free; keyed: paid band) |
| Perplexity | Anonymous or `PERPLEXITY_API_KEY` | Automatic (best-effort; keyed: paid band) |
| DeepSeek | Model login or `DEEPSEEK_API_KEY` | Automatic (paid) |
| Brave | `BRAVE_API_KEY` | Automatic (paid) |
| SerpApi | `SERPAPI_API_KEY` | Automatic (paid) |
| SearXNG | Self-hosted endpoint URL | Automatic (free, with an endpoint) |

The same controls are available in-app via `/search`.

## Skills and guides

Drop a `SKILL.md` (with optional reference files) into
`~/.local-operator/skills/<name>/` and the agent indexes it semantically.
Only the skills relevant to the current turn are surfaced, and their bodies
load on demand via `skill://<name>` reads, so your context isn't taxed by
knowledge you aren't using. `/skills` lists what's loaded.

Skills can also live in grouped libraries: `~/.local-operator/skills/<group>/<name>/SKILL.md`
is discovered too (up to 3 levels below a skills root by default; set
`LOCAL_OPERATOR_SKILL_MAX_DEPTH` to 1-5 to change it, `1` = flat only). A
directory that holds a `SKILL.md` is a skill boundary and is never searched
further, so a skill's own `references/` or `scripts/` folders never register as
skills.

You can also invoke one **by name** instead of leaving the choice to the
router: type `$` in the composer to pick from your skills, then write the
request: `$research compare these two API designs` loads that skill and
sends it with your message. A skill marked `hide: true` is excluded from
automatic routing and is reachable this way only, which makes `$` the home
for "never pick this on your own, but run it when I say so".

## MCP servers

Local Operator speaks [MCP](https://modelcontextprotocol.io/) over the
official SDK, with lazy tool loading: servers advertise a bounded summary, and
individual tool schemas enter the context only when the agent actually enables
them.

```bash
lop mcp add linear --url https://mcp.linear.app/mcp --oauth
lop mcp login linear     # complete the OAuth flow
lop mcp list
```

Server configs are discovered from the project (`.local-operator/mcp.json`,
`.mcp.json`), your home directory, and best-effort imports of Claude Code,
Cursor, VS Code, and Codex CLI configs, so servers you already configured
elsewhere are available without re-declaring them. See
[docs/mcp.md](../../docs/mcp.md) for the trust model before enabling
project-supplied servers.
