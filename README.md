<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./static/local-operator-icon-2-dark-clear.png">
  <source media="(prefers-color-scheme: light)" srcset="./static/local-operator-icon-2-light-clear.png">
  <img alt="Local Operator logo"
       src="./static/local-operator-icon-2-light-clear.png">
</picture>

<h1 align="center">Local Operator</h1>
<div align="center">
  <h3>An open-source AI agent that lives in your terminal and works on your machine</h3>
  <p><i>Plans, runs tools, browses, spawns subagents, and remembers — all from a fast terminal UI</i></p>
</div>

<br />

<p align="center">
  <img src="./static/tui-hero.png" alt="Local Operator TUI running a real task: streamed response, an expanded tool card showing a command and its output, and a live status line" width="720">
</p>

<p align="center"><i>The Local Operator TUI mid-task: streamed responses, expandable tool cards, and one-line receipts for everything the agent does.</i></p>

<br />

**Local Operator** is a terminal-native AI agent: describe what you want done
and it does the work on your machine, asking before anything writes or
executes. It is MIT-licensed and built to be lived in — sessions persist and
resume, context compacts itself before it overflows, and the agent can
schedule its own follow-ups.

<div align="center">
  <a href="#-quickstart">Quickstart</a> •
  <a href="#️-a-tour-of-the-tui">Tour</a> •
  <a href="#-providers">Providers</a> •
  <a href="#️-headless--server-modes">Headless & Server</a> •
  <a href="#-contributing">Contribute</a>
</div>

## 📚 Table of Contents

- [✨ Why Local Operator](#-why-local-operator)
- [🚀 Quickstart](#-quickstart)
- [🖥️ A Tour of the TUI](#️-a-tour-of-the-tui)
  - [Slash commands](#slash-commands)
  - [Keys worth knowing](#keys-worth-knowing)
- [🔌 Providers](#-providers)
- [🧰 What the Agent Can Do](#-what-the-agent-can-do)
  - [🔎 Web search](#-web-search)
  - [🤝 Subagents, agent profiles, and teams](#-subagents-agent-profiles-and-teams)
  - [🧠 Skills and guides](#-skills-and-guides)
  - [🔗 MCP servers](#-mcp-servers)
- [⚙️ Headless & Server Modes](#️-headless--server-modes)
- [📦 Installation Options](#-installation-options)
- [🔧 Configuration & Credentials](#-configuration--credentials)
- [🌟 Radient Agent Hub](#-radient-agent-hub)
- [🔒 Safety Model](#-safety-model)
- [📝 Examples](#-examples)
- [👥 Contributing](#-contributing)
- [📜 License](#-license)

## ✨ Why Local Operator

- **A real terminal UI, not a REPL.** A full-screen [Textual](https://textual.textualize.io/)
  app with streamed responses, expandable tool cards, session resume, themes,
  and a status line that tells you what the agent is doing and what it costs.
- **Sign in with the account you already have.** OAuth login for OpenAI
  (ChatGPT), Anthropic (Claude), Kimi, xAI, Z.AI, and Qwen — or bring an API
  key, or run entirely offline with Ollama.
- **Approval-gated execution.** Reads are automatic; writes and shell commands
  ask first. `/approvals auto` (or `--yolo`) opts out deliberately, per
  session or as a saved default.
- **Parallel work.** The agent fans out background subagents for independent
  slices, keeps working while they run, and reports back — you can peek at,
  steer, or cancel any of them. Reusable agent profiles and multi-agent teams
  are built in.
- **A session you can leave and come back to.** Transcripts persist,
  `/resume` picks up where you left off, and context compaction runs itself
  before the window fills, so long sessions don't fall off a cliff.
- **Skills, MCP, scheduled wakes, web search, a browser tool** — the agent's
  toolbox is broad, and everything it does leaves a visible receipt in the
  transcript.

## 🚀 Quickstart

Requires Python 3.12+.

```bash
pip install local-operator     # pipx install local-operator on Linux (PEP 668)
```

Sign in to a provider (or skip this — the app tells you what's missing on
first run):

```bash
local-operator login           # lists login-capable providers
local-operator login anthropic # OAuth sign-in in your browser
```

Then start it:

```bash
local-operator
```

That's it. Type what you want done. `esc` stops the agent, `/help` lists
commands, `/exit` quits.

<p align="center">
  <img src="./static/tui-welcome.png" alt="The Local Operator welcome screen with rotating tips and the composer ready for a first prompt" width="720">
</p>

Prefer a fully local model? Install [Ollama](https://ollama.com/download),
pull a model, and point the agent at it:

```bash
local-operator --hosting ollama --model qwen2.5:14b
```

## 🖥️ A Tour of the TUI

Everything the agent does shows up as a card or a one-line receipt. Tool
cards expand (`enter`/`space`) to show the full command and output; the
status line tracks the current step, token usage, and cost.

When a tool call needs your sign-off, the approval prompt shows exactly what
is about to run before anything touches your system:

<p align="center">
  <img src="./static/tui-approval.png" alt="An approval prompt showing the exact shell command awaiting user confirmation" width="720">
</p>

Switching models is a picker, not a config file — `/model` lists every model
your signed-in providers offer, with fuzzy filtering:

<p align="center">
  <img src="./static/tui-model-picker.png" alt="The /model picker filtering across providers" width="720">
</p>

Ask for parallel work and the agent delegates: the subagent dock shows each
worker's status live, and you can open any of them to watch its transcript:

<p align="center">
  <img src="./static/tui-subagents.png" alt="The subagent panel with several background workers running concurrently" width="720">
</p>

Coming back later is `/resume` — a picker over your recent sessions, each
with its title and age:

<p align="center">
  <img src="./static/tui-resume.png" alt="The /resume session picker listing recent conversations" width="720">
</p>

And `/usage` answers the question every agent user has: how much quota is
left, and what did this session cost?

<p align="center">
  <img src="./static/tui-usage.png" alt="The /usage panel showing provider quota and per-session spend" width="720">
</p>

### Slash commands

`/help` shows the full table in-app. The highlights:

| Command | What it does |
| --- | --- |
| `/model` | Switch model for this session; `/model default` saves it for new ones |
| `/effort` | Show or set reasoning effort (`shift+tab` cycles) |
| `/approvals` | Set whether tools ask first (`ask`/`auto`; add `default` to keep it) |
| `/resume` | Pick a past conversation and continue it |
| `/new`, `/clear`, `/reload` | Fresh conversation · wipe the screen · reboot the session in place |
| `/goal`, `/loop` | Set an objective, then iterate autonomously toward it |
| `/btw` | Ask a side question off the record — it never joins the conversation |
| `/compact` | Compact the context now (it also happens automatically) |
| `/usage`, `/context` | Provider quota and spend · what's occupying the context window |
| `/provider`, `/login`, `/logout`, `/accounts`, `/credential` | Manage providers and stored credentials |
| `/search` | Configure web-search providers and load balancing |
| `/skills`, `/mcp`, `/team` | List loaded skills · MCP servers · manage teams |
| `/theme`, `/rename` | Pick from 20+ built-in themes (arrows preview live) · rename the session |

### Keys worth knowing

- **Type while the agent works** — your message is delivered at the next
  step as steering, no need to wait.
- `esc` — stop the agent without ending the session.
- `ctrl+b` — open an aside (side question) without losing what you were typing;
  `ctrl+f` promotes the aside into the conversation.
- `shift+tab` — cycle reasoning effort.
- `ctrl+l` — clear the transcript (history is untouched).

## 🔌 Providers

One agent, your choice of brain. OAuth providers sign in through the browser
and use your existing subscription; API-key providers prompt once and store
the key locally; Ollama runs models on your own hardware.

| Provider | Access |
| --- | --- |
| OpenAI / ChatGPT | OAuth (browser or device code) or `OPENAI_API_KEY` |
| Anthropic / Claude | OAuth or `ANTHROPIC_API_KEY` |
| Kimi (Moonshot) | OAuth or `KIMI_API_KEY` |
| xAI / Grok | OAuth or `XAI_API_KEY` |
| Z.AI (GLM) | OAuth or API key |
| Qwen (Alibaba) | OAuth (token plan) or API key |
| Google Gemini | `GOOGLE_AI_STUDIO_API_KEY` |
| DeepSeek | `DEEPSEEK_API_KEY` |
| Mistral | `MISTRAL_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` — one key, many models |
| Radient | `RADIENT_API_KEY` — automatic per-step model selection |
| Ollama | Local, no key, no network |

```bash
local-operator login              # list login-capable providers
local-operator login openai       # OAuth flow
local-operator login-status       # what's signed in
local-operator logout kimi
```

Legacy `--hosting <name> --model <name>` flags keep working, and API keys can
be set non-interactively with `local-operator credential update <KEY_NAME>`.

## 🧰 What the Agent Can Do

The agent's built-in tools, each with its own card in the transcript:

- **Run things** — `bash` (shell commands), `eval` (a persistent Python
  kernel: variables survive across calls).
- **Work with files** — `read`, `write`, `edit` (surgical search/replace),
  `glob`, `grep`, plus `lsp` for language-server-backed code intelligence.
- **Reach the web** — load-balanced `web_search` across seven providers and a
  `browser` tool for pages that need rendering or interaction.
- **Stay organized** — a visible `todo` list for multi-step work, `ask` to
  put real decisions back to you as a picker instead of a wall of text.
- **Work in the background** — `task` spawns subagents, `jobs`/`wait`/`hub`
  manage and talk to them, and `wake` schedules future follow-ups
  ("check the build again in 30 minutes").

### 🔎 Web search

Search works out of the box — DuckDuckGo and Tavily's keyless endpoint are
enabled by default, and requests rotate across providers with automatic
fallback when one is rate-limited or down:

```bash
local-operator search list
local-operator search test "Python 3.13 release notes"
local-operator search enable perplexity
local-operator search setup brave --api-key
local-operator search setup tavily --oauth      # official Tavily MCP server
local-operator search setup searxng --endpoint https://search.example.com
```

| Provider | Access | Default |
| --- | --- | --- |
| DuckDuckGo | Credential-free | Enabled |
| Tavily | Keyless, `TAVILY_API_KEY`, or OAuth MCP | Enabled |
| Perplexity | Anonymous or `PERPLEXITY_API_KEY` | Disabled |
| Brave | `BRAVE_API_KEY` | Disabled |
| Exa | `EXA_API_KEY` | Disabled |
| SerpApi | `SERPAPI_API_KEY` | Disabled |
| SearXNG | Self-hosted endpoint URL | Disabled |

The same controls are available in-app via `/search`.

### 🤝 Subagents, agent profiles, and teams

Ask for parallel work and the agent splits it into concurrent background
subagents, each with a role (`reviewer`, `coder`, `scout`, …) that carries
vetted guidance. You can watch, steer, pause, or cancel any of them from the
TUI's subagent panel.

Roles and specialists are reusable **agent profiles** (`agent` tool), and a
manager-plus-roster **team** can be saved and launched by name. Agents can
also be managed from the CLI:

```bash
local-operator agents create "My Agent"
local-operator agents list
local-operator teams list
```

### 🧠 Skills and guides

Drop a `SKILL.md` (with optional reference files) into
`~/.local-operator/skills/<name>/` and the agent indexes it semantically —
only the skills relevant to the current turn are surfaced, and their bodies
load on demand via `skill://<name>` reads, so your context isn't taxed by
knowledge you aren't using. `/skills` lists what's loaded.

### 🔗 MCP servers

Local Operator speaks [MCP](https://modelcontextprotocol.io/) over the
official SDK, with lazy tool loading: servers advertise a bounded summary,
and individual tool schemas enter the context only when the agent actually
enables them.

```bash
local-operator mcp add linear --url https://mcp.linear.app/mcp --oauth
local-operator mcp login linear     # complete the OAuth flow
local-operator mcp list
```

Server configs are discovered from the project (`.local-operator/mcp.json`,
`.mcp.json`), your home directory, and best-effort imports of Claude Code,
Cursor, and VS Code configs — so servers you already configured
elsewhere just show up. See [docs/mcp.md](./docs/mcp.md) for the trust model
before enabling project-supplied servers.


## ⚙️ Headless & Server Modes

**One-shot execution** for scripts and automation:

```bash
local-operator exec "summarize the failures in ./test.log"
local-operator exec "long migration" --background   # detach with a log file
local-operator exec "audit deps" --json             # one JSON line per event
```

Exit code 0 on success — pipeline-friendly.

**Server mode** exposes the agent as a FastAPI service (used by the optional
[desktop UI](https://github.com/damianvtran/local-operator-ui)):

```bash
pip install "local-operator[server]"
local-operator serve                 # http://localhost:1111, docs at /docs
```

**Phone access** — an optional mobile portal daemon lets you check on and
steer sessions from your phone:

```bash
local-operator mobile install
local-operator mobile status
```

## 📦 Installation Options

The default install is deliberately small. Optional features live behind
extras:

| Extra | Adds |
| --- | --- |
| `server` | The HTTP API server (`local-operator serve`) and background scheduler |
| `mcp` | Model Context Protocol client support |
| `images` | HEIC/HEIF image attachment decoding |
| `tokenizer` | Exact BPE token counting (estimated otherwise) |
| `all` | Everything above |

```bash
pip install "local-operator[all]"    # quote it — shells glob the brackets
```

If you hit a feature whose extra is missing, the agent tells you which one to
install instead of failing with an import error.

**Nix**: `nix develop` drops you into a reproducible dev shell via the
provided `flake.nix`.

**Docker**: `docker compose up -d` with the provided compose file.

## 🔧 Configuration & Credentials

Configuration lives at `~/.local-operator/config.yml`:

```bash
local-operator config create      # scaffold it
local-operator config list        # every option, with descriptions
local-operator config edit <key> <value>
local-operator config open        # open it in your editor
```

Commonly set values: `hosting` and `model_name` (skip the CLI flags),
`conversation_length` / `detail_length` (history kept verbatim vs
summarized), and `tui.theme` (any registered theme name — easier to set with
`/theme`, which previews live).

Credentials are stored in `~/.local-operator/credentials.env` and never
echoed:

```bash
local-operator credential update TAVILY_API_KEY
local-operator credential delete TAVILY_API_KEY
```

OAuth tokens from `local-operator login` are stored separately and refresh
themselves.

## 🌟 Radient Agent Hub

[Radient](https://console.radienthq.com) adds two optional capabilities:

- **Automatic model selection** — `local-operator --hosting radient` picks
  the best model per step to balance quality and cost, no `--model` needed.
- **Agent sharing** — push your agents to the public hub, pull agents others
  published:

```bash
local-operator credential update RADIENT_API_KEY
local-operator agents push --name "My Agent"
local-operator agents pull --id "<agent_id>"     # no key needed to pull
```

## 🔒 Safety Model

- **Approval tiers.** Read-only tools run automatically; anything that writes
  files or executes commands prompts first, showing the exact command.
  `/approvals auto` or `--yolo` disables prompts only when you say so.
- **Visible receipts.** Every tool call leaves a card or one-line receipt in
  the transcript — there is no invisible action.
- **Local-first options.** Run Ollama models for closed-circuit operation
  where nothing leaves your machine.
- **MCP trust model.** Project-supplied MCP configs are treated as trusted
  input and warned about on first connect — see [docs/mcp.md](./docs/mcp.md).
- **Credential hygiene.** Keys live in a local credential store, are entered
  through hidden prompts, and are kept out of transcripts.

## 📝 Examples

👉 The [example notebooks](./examples/notebooks/) show real tasks completed
with Local Operator, saved from live sessions:

- 🔄 **[Automated commit message generation](examples/notebooks/github_commit.ipynb)** from git diffs
- 🔀 **[End-to-end pull request automation](examples/notebooks/github_pr.ipynb)** — creation, review, template completion
- 🔢 **[MNIST digit recognition](examples/notebooks/kaggle_digit_recognizer.ipynb)** — 99.3% accuracy on the Kaggle competition
- 🏠 **[House price prediction with XGBoost](examples/notebooks/kaggle_home_data_competition.ipynb)** — top 5% Kaggle score
- 🚢 **[Titanic survival prediction](examples/notebooks/kaggle_titanic_competition.ipynb)** with LightGBM
- 🌐 **[Web research and data extraction](examples/notebooks/web_research_scraping.ipynb)** — scraping a sanctions list
- 📈 **[Business pricing analysis](examples/notebooks/business_pricing_margin.ipynb)** — optimal subscription pricing

## 👥 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for how to
submit bug reports and feature requests, set up a development environment,
and open pull requests. `docs/` covers the architecture
([REWRITE.md](./docs/REWRITE.md)), benchmarks, and verification evidence.

## 📜 License

MIT — see [LICENSE](LICENSE) for details.
