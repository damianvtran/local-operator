<!-- Moved verbatim from the project README (2026-11-08 README refresh); the README now links here. -->

# Installation options and configuration

## Installation Options

The default install includes the TUI, the server, MCP and the other
features most people use; a few heavier or niche ones live behind extras:

| Extra | Adds |
| --- | --- |
| `server` | The HTTP API server (`lop serve`) and background scheduler |
| `mcp` | Model Context Protocol client support |
| `images` | HEIC/HEIF image attachment decoding |
| `tokenizer` | Exact BPE token counting (estimated otherwise) |
| `fetch` | Markdownify renderer for `web_fetch` |
| `lsp` | Jedi-backed symbol-aware Python navigation for the `lsp` tool |
| `all` | Everything above except `lsp` |

```bash
pip install "local-operator[all]"    # quote it, or your shell globs the brackets
```

If you hit a feature whose extra is missing, the agent tells you which one to
install instead of failing with an import error.

**Nix**: `nix develop` drops you into a reproducible dev shell via the
provided `flake.nix`.

**Docker**: `docker compose up -d` with the provided compose file.

## Configuration & Credentials

Configuration lives at `~/.local-operator/config.yml`:

```bash
lop config create      # scaffold it
lop config list        # every option, with descriptions
lop config edit <key> <value>
lop config open        # open it in your editor
lop config instructions  # which instruction files a session assembles, in order
```

Commonly set values: `hosting`, `model_name` and `model_effort` (skip the CLI
flags; `model_effort` sets the reasoning level new conversations start at, and a
rung the chosen model lacks clamps to its nearest),
`conversation_length` / `detail_length` (history kept verbatim vs
summarized), `tui.theme` (any registered theme name, easier to set with
`/theme`, which previews live), and `retry.fallbackChains` (the model
cascade). See [Subscriptions](subscriptions.md#subscriptions). `bash.shell` picks the
interpreter the `bash` tool spawns: unset, it runs the first `bash` on `PATH`
(Homebrew bash 5 when installed, else the system one) and falls back to
`/bin/sh` only on a host with no bash, so process substitution and other bash
syntax work as the tool's name promises. Every option is browsable in
`/settings`.

### Credentials and secrets

Five different things get called "credentials" here, and they behave
differently. **Provider API keys** are the keys `lop` itself reads: they live
in `~/.local-operator/credentials.env` as plaintext, and you set them once
through a masked prompt. **Provider logins** are the OAuth tokens `lop login`
gets, stored in `~/.local-operator/auth.db`, where they refresh themselves. The
QwenCloud console ticket is one login you keep by hand (below).
**Session credentials** are what a secret handed over through the `/credential`
capture becomes: held in memory for that session, injected into every `bash`
child's environment so a command can use them, and never readable by the model.
**Ordinary variables** are the named values a session carries for the agent,
shared through the `list_variables` and `read_variable` tools; they are not
secret at all, and they are filtered against a list of names that must never be
passed on. The **encrypted long-term store** (`~/.local-operator/secrets/`) is
where a secret goes when it has to outlive the session, and it is the main
subject of this section.

`lop credential` stores provider API keys; `lop secret` stores every other
secret you save yourself, the QwenCloud ticket included.

```bash
lop credential update OPENROUTER_API_KEY    # provider key, masked prompt
lop credential delete OPENROUTER_API_KEY
```

**What the `lop secret` store is, and is not.** Secrets are encrypted with
AES-256-GCM under names the store keeps only as a keyed hash, so the store's own
directory holds no readable `.env`, no greppable `API_KEY=`, and not even the
secret names. That is a statement about the store alone: the provider keys above
are still a plaintext file.

In the default mode the master key is a file beside the store, so anyone who
copies that directory can decrypt it. Against an attacker who reads the key
file itself, that is the same bar as a plaintext `.env`. `lop secret harden`
moves the store to passphrase mode, where the only key on disk is a
passphrase-wrapped copy and the unwrapped master key lives in the memory of a
running broker; it is not the passphrase that stays in memory, and reading that
key from another process raises a macOS authorization prompt. Run
`lop secret unlock` once after each boot to put the key back in the broker.

What the store is **not** is a vault: `lop` is on your `PATH`, so anything
running as you that is willing to run `lop` can read every secret exactly as
the agent does, and a value an agent retrieves is in that session's memory for
the turn. `lop secret status` reports the mode, and the risk that comes with it
when the command can open the store: the key file itself in the default mode,
the unlocked broker's memory in passphrase mode. When it cannot open the store
it prints the reason and the command that fixes it, and no note. Neither mode
is a vault, and the limit above holds in both. The store raises the cost of
stealing these credentials a long way. It is not a guarantee.

**One provider login you keep by hand** is the QwenCloud console ticket: a
browser session cookie you capture and store yourself, expiring roughly weekly.
Its value goes into the encrypted `lop secret` store rather than into `auth.db`,
which keeps only the ticket's metadata: its name, when it was captured, and
its length. The
[QwenCloud guide](../../local_operator/guides/qwencloud/GUIDE.md) covers it,
including `lop qwencloud-ticket migrate` for a ticket stored by an older build.

The `lop secret` commands are the main way in. Values are read from stdin,
never from the command line, where any process running as you can read them:

```bash
printf %s "$TOKEN" | lop secret set GITHUB_TOKEN --description "CI token"
lop secret list                     # names and descriptions, never values
lop secret get GITHUB_TOKEN         # value to stdout, no trailing newline
lop secret rm GITHUB_TOKEN
```

Use a stored secret in the command that needs it:

```bash
curl -H "Authorization: Bearer $(lop secret get GITHUB_TOKEN)" https://api.github.com/user
```

The value reaches `curl` inside the child process; the model sees only the
command text it wrote, and the value never enters the transcript.
Neither you nor the agent should ever echo a secret, print a variable holding
one, write one to a file you do not immediately delete, or paste one into a
commit or a message, because each of those puts it somewhere it outlives the
command.

**From the agent's side**, the `secret` tool takes
`op: store|retrieve|list|describe|update|delete`. `retrieve` deliberately
returns a receipt naming the secret and how to reach it, never the value: in
`bash` that is `$(lop secret get NAME)`, and in the `eval` kernel it is
`secrets["NAME"]`, which hands the real string to the worker process rather
than to the transcript. Agent-facing guidance lives in
[`local_operator/guides/credentials/GUIDE.md`](../../local_operator/guides/credentials/GUIDE.md).

**Type `/credential` and a space, then type the secret, or paste it so your
terminal delivers it as text**, and it is captured as a credential chip. Typing
is masked and captured. So is a paste your terminal delivers as text: `Cmd+V` on
macOS, or your terminal's own Paste command elsewhere. The composer's own paste
key, `Ctrl+V` on macOS, is **not** captured: it reads the clipboard directly and
inserts the value as ordinary text, so a secret must not be pasted that way. The
value goes to this session's credential store, the line
carries only the chip, and you keep typing to say what the secret is for: that
description, not the secret, is what the agent reads.

<p align="center">
  <img src="../../static/tui-credential-chip.png" alt="A session's composer showing the line Here's my OpenRouter key, use it to test the integration followed by a collapsed [Credential #1, 15 chars] chip" width="720">
</p>

<p align="center"><i>A key handed over with /credential goes to this session's credential store: the composer keeps a chip in its place, and your description is what the agent reads.</i></p>

**Without the `/credential` token, a paste is not captured.** What opens the
capture is the `/credential` token, not what the text looks like: paste a key
into a composer that has not been armed and it is inserted exactly as typed,
into the draft, into the prompt and on to the provider. This gesture is a
hand-over, not a safety net.

**Typing into the capture is masked as well.** With the token entered, the
characters are held out of the draft and painted as dots, and **Enter** turns
them into the same chip. **Esc** cancels the capture and puts the characters
back in the line as plain text. Bare `/credential` lists what this session
holds, and `/credential --persist <KEY>` also saves a session credential to the
long-term store.

<p align="center">
  <img src="../../static/tui-credential-masked-typing.png" alt="The composer while a secret is being typed after /credential: the characters render as dots, with the notice underneath reading masked as you type — Enter turns it into a chip, Esc cancels" width="720">
</p>

<p align="center"><i>A hand-typed secret is masked as you type: Enter turns it into a chip, Esc cancels the capture and puts your text back in the line.</i></p>

### What the agent's shell may see

`bash` and the `eval` kernel are real child processes, and both take their
environment from the session's `shell_environment` policy:

| `mode` | What a command the model runs can see |
| --- | --- |
| `inherit` (default) | a copy of the environment `lop` was started with, so `gh`, `aws`, `npm`, your proxy settings and your locale all keep working |
| `allowlist` | `HOME`, `LOGNAME`, `PATH`, `SHELL`, `TERM`, `USER`, whatever `shell_environment.inherit` names, and nothing else |

**What the strict mode does and does not close.** It closes the *inherited
copy*: the provider key `lop` was started with is no longer handed to the child,
and the child's own environment becomes a decision this program makes rather
than an accident of how it was launched. It does **not** make the key
unreadable. A command the model writes runs as the same user as `lop`, so it can
read the parent process's environment in one line — `ps eww -p $PPID` on macOS,
`cat /proc/$PPID/environ` on Linux. Read `allowlist` as "the commands my agent
runs no longer carry the credential", not as "the credential cannot be recovered
from a command". Closing that second half means not holding the provider key in
the process environment at all (or running the child as another user), which is
a separate change.

**The mode is resolved once per session.** `config.yml` lives in a directory the
agent's own shell can write, as the same user, so a policy re-read per command
is one the model could turn off between two commands. An edit here applies at
the next launch, and the settings page marks the section accordingly. The
consequence to know: if a run can rewrite its own config *before* its first
command, treat the strict mode as a mitigation rather than a guarantee.

The `bash` child keeps the harness's own additions in either mode: the
non-interactive defaults (pagers off, `CI=1`, `LOCAL_OPERATOR_AGENT_SHELL=1`,
`TERM=dumb`) and any credential the session store injected for the agent to
*use*. The `eval` worker is never handed
those: it gets its own protocol channel instead, so `lop` gives a cell no `CI`
and no session credential in either mode. It does still see whatever the harness
was *launched* with when the mode is `inherit` (a `CI=1` the launching
environment exported included) — the difference is the injection, which is a
`bash`-only addition. Any subprocess the cell spawns follows the same policy. In either mode
`shell_environment.exclude` removes names outright — including those injections,
so it is the way to deny a variable rather than merely not grant it. It governs
these two children only: MCP servers (already limited to the SDK's safe set plus
their own configured `env`) and `lop`'s own processes are unaffected. A name in
`inherit` or `exclude` that matches nothing is a no-op and is reported — a
denial at `WARNING`, a grant at debug — so a typo cannot pass for a control.

```yaml
values:
  shell_environment:
    mode: allowlist                 # strict; the default is `inherit`
    inherit: [LANG, LOCAL_OPERATOR_CONFIG_DIR]   # names the strict mode keeps
    exclude: [GH_TOKEN]             # never passed on, in either mode
```

Set it per deployment, not globally: the interactive session you are sitting at
wants `inherit`, a server-owned run wants `allowlist`. Why the strict mode
exists: `lop` reads its provider API key **from its own environment** before the
copy in `credentials.env`, and falls back to that file only when the environment
does not supply a value for the key. The file is not consulted for every
provider: a request reads Anthropic's key from the environment or a stored
login, and never from `credentials.env`. An inherited copy is the copy a request
uses when the provider has no stored login, and it is a spend credential in the
hands of any command the model writes. A run whose whole job is fetching
attacker-influenceable pages is exactly the run that must not have one. `mode`
unset (or blank) is `inherit`;
an unrecognised value (`strict`, `allow-list`) resolves to `allowlist` and warns
by name, so a typo cannot silently disarm a hardened run. Every name here is a
variable NAME, never a value.

### Standing instructions

Your machine-wide preferences — output style, commit conventions, safety
rules — live in `~/.local-operator/system_prompt.md`, editable from
Settings → Instructions, `PATCH /v1/config/system-prompt`, or your editor.
They ride every session and every subagent.

If you also run Claude Code, Codex, opencode or droid, Local Operator reads
`~/.agents/AGENTS.md` too — the tool-neutral file those ecosystems have
converged on, in the directory Local Operator already scans for skills. It is
**read-only** (`system_prompt.md` stays the only file Local Operator writes),
it is placed *before* your own instructions so those win on conflict, and
content identical to `system_prompt.md` is dropped rather than sent twice.

```bash
# read a different set of shared files instead (colon-separated)
export LOCAL_OPERATOR_ECOSYSTEM_INSTRUCTIONS=~/.config/AGENTS.md:~/team/AGENTS.md
# or turn the import off entirely
export LOCAL_OPERATOR_ECOSYSTEM_INSTRUCTIONS=
```

`lop config instructions` reports which files a session actually assembles, in
order, with the size each contributed and whether it was collapsed as a
duplicate — paths and sizes only, never the contents.

Project-level `AGENTS.md` / `CLAUDE.md` files are discovered separately by
walking up from your working directory, and can be disabled with
`LOCAL_OPERATOR_CONTEXT_FILES=0`. Inside a git repository under your home
directory, the walk continues past the repository root to the folder just
below home, so a folder that groups several repos can carry their shared
guidance; the guidance file in home itself is not included. A repository
under a dot-named folder such as `~/.codex`, or itself dot-named, still stops
at its own root.
