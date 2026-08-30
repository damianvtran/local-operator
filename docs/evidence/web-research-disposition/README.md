# Evidence: research disposition (web_search / web_fetch)

Captured 2026-08-29 on macOS 25.6.0, against `origin/main` @ `a892aaf5` (before)
and `dev-web-research-disposition` (after). The behavioural arm uses live
`anthropic/claude-opus-5` turns on the configured OAuth credential; web results
come from the credential-free `duckduckgo` provider.

| File | What it shows |
|---|---|
| `token-cost.txt` | Always-on prompt cost before vs after, on `bench_role_overhead.py`'s basis |
| `cache-stability.txt` | Prompt-cache prefix stability before vs after (unchanged) |
| `ab-eval.txt` | Live A/B: research rate over 14 scenarios x 3 runs x 2 arms |
| `ab-eval.json` | The same run as structured data, including every tool call made |
| `token_probe.py` | The snippet that produces `token-cost.txt`, so the numbers are re-derivable |

Re-derivable without a paid run except the last two:

```sh
.venv/bin/python docs/evidence/web-research-disposition/token_probe.py
.venv/bin/python scripts/bench_cache_rate.py --turns 4
.venv/bin/python scripts/eval_research_disposition.py \
    --before <worktree-of-base> --after <worktree-of-branch> --runs 3
```

## The problem this measures

`system.md` never mentioned `web_search` or `web_fetch` — zero occurrences of
"web", "internet", "research" or "look up" in 207 lines. The tools were not
gated: both are in `DEFAULT_TOOL_NAMES` and enabled by default. They were
present and unmentioned.

Worse than unmentioned, they were implicitly excluded. Two places defined tool
use as a closed set of local actions:

- the persona paragraph — "tools for running shell commands, reading and
  editing files, searching the workspace... Use them whenever the answer
  depends on **the machine's state**";
- the first working principle — "Verify with a tool rather than assuming: run
  the command, read the file, **search the workspace**."

Both enumerations are complete-sounding and entirely local. An agent following
them faithfully never reaches the web, because by the definition it was handed
it has already verified. That is why lookups only happened when a user asked
for one explicitly. The fix widens both enumerations and adds one bullet naming
the trigger; it does not advertise the tools.

## Why the cost is not in the tool descriptions

The obvious fix — explain "when to reach for it" in the two tool descriptions —
is the expensive one. A tool's description is billed **twice**: once in the
provider `tools` array, and again in the rendered tool inventory, because
`_render_tool_inventory` emits the full description string into system block
`[1]`. 36 tokens of prose there costs 144 always-on tokens across the two
tools, for guidance read at call-selection time rather than at task-framing
time. The behaviour wanted here is *noticing that a lookup is needed*, which
happens before the model is scanning tool schemas.

`token-cost.txt` shows `tool_inventory` and `tools_array` both unchanged.

## What the A/B eval measures, and the two traps in measuring it

`ab-eval.txt` reports two rates per arm, and the second one matters as much as
the first:

- **SHOULD-trigger** (10 scenarios): questions whose answer is not on this
  machine — a dependency error message, a published advisory, a version-specific
  API, current ecosystem practice, design patterns.
- **SHOULD-NOT-trigger** (4 scenarios): a local refactor, a question about lop
  itself, arithmetic, a directory listing. Prompting an agent to search is
  trivial; the failure it buys is a reflexive search on every task, a latency
  and token cost paid on work that never needed it. S12 ("how do I change my
  custom instructions in lop") is the highest-signal negative: the correct
  behaviour is `read guide://configuration`, so a web search there is a
  regression against an existing rule, not merely a wasted call.

Two measurement traps were hit and fixed while building this, both of which
would have produced a confident wrong number:

1. **Counting only `web_search`/`web_fetch` understates research.** An agent
   can reach the network by curling from `bash` or opening a URL from `eval`,
   and on the first real run *both* arms queried the OSV advisory API that way.
   That is still research. The harness therefore scores "researched" (network
   reached at all) and reports "via web tool" separately — moving a run from
   improvised curl to the purpose-built tool is a real improvement (bounded
   output, spill handles, caching, provider fallback), but it is not the same
   claim as creating research behaviour that was not there.

2. **The arms must be isolated from this machine.** Both run against a clean
   `LOCAL_OPERATOR_CONFIG_DIR` seeded with credentials only, with ecosystem
   skill roots disabled. The operator's own `system_prompt.md` and skills do
   not ship to users, and either could produce search behaviour on its own and
   have it credited to the prompt change.

   **The MCP half of that isolation does not work, and the runs recorded here
   carry it.** `_prepare_home` writes an empty `mcp.json` into the scratch
   config dir, but `resolve_mcp_config` reads `Path.home() / ".local-operator"
   / "mcp.json"` directly (`local_operator/mcp/config.py:306`) and never
   consults `LOCAL_OPERATOR_CONFIG_DIR`. So the operator's own MCP servers
   (Notion, HubSpot, Slack, Datadog) were connected in all 84 runs, and their
   tool definitions rode both arms' prompts.

   Scope of the damage, checked rather than assumed: the leak is **symmetric**
   (42 runs per arm) and **no `mcp__*` tool is called in any of the 84 runs** —
   the only tools that appear are `bash`, `eval`, `glob`, `grep`, `read`,
   `web_fetch`, `web_search`. The A/B contrast therefore stands. What is *not*
   safe to read off these runs is any absolute prompt size, because both arms
   carried MCP surface the `token-cost.txt` baseline does not include.
   `token-cost.txt` is measured separately and offline, so it is unaffected.

   The harness has since been fixed: it redirects `HOME` at the scratch dir as
   well as `LOCAL_OPERATOR_CONFIG_DIR`, which covers both the code that calls
   `config_dir()` and the code that reads `Path.home()` itself. Verified on a
   spot run — no MCP traffic in either arm. The recorded `ab-eval.json` predates
   that fix and is kept as captured rather than silently re-run, since the
   contrast it reports is unaffected.

The harness also fingerprints each arm's prompt text at start and end and
refuses to report a run where it changed mid-flight — a full run takes over an
hour and reads the prompt from the working tree on every scenario, so an edit
during the run silently averages two different texts into one number.
