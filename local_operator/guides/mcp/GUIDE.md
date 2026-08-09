---
name: mcp
summary: Install and configure MCP servers.
description: Install, configure, list, troubleshoot, and remove Model Context Protocol (MCP) servers in Local Operator.
---

# MCP servers in Local Operator

Use Local Operator's MCP commands instead of editing merged configuration unless a server needs options the CLI does not expose.

## Add a server

Stdio server:

```bash
local-operator mcp add filesystem \
  --command npx \
  --arg -y \
  --arg @modelcontextprotocol/server-filesystem \
  --arg /allowed/path
```

Remote HTTP or SSE server:

```bash
local-operator mcp add example --url https://example.com/mcp
```

For a hosted server that uses OAuth, mark it explicitly and complete the
interactive login. The browser eventually redirects to a loopback URL; if the
page cannot connect, paste that full URL back into the waiting terminal. Local
Operator stores the resulting token in its credential database and reuses it
in future sessions:

```bash
local-operator mcp add linear --url https://mcp.linear.app/mcp --oauth
local-operator mcp login linear
```

Add `--scope project` to write `<cwd>/.local-operator/mcp.json`; the default global scope writes `~/.local-operator/mcp.json`. Use repeated `--env KEY=VALUE` only when the server must receive that variable. Do not put long-lived secrets in committed project configuration.

Check and remove entries:

```bash
local-operator mcp list
local-operator mcp remove filesystem
local-operator mcp remove filesystem --scope project
```

Start a new Local Operator session after changing MCP configuration. MCP tools are discovered and connected during session startup; their model-visible names use `mcp__<server>_<tool>`.

## Configuration discovery and precedence

Highest precedence first:

1. `<cwd>/.local-operator/mcp.json`
2. `<cwd>/.mcp.json`
3. `~/.local-operator/mcp.json`
4. Compatible imports from Claude, Cursor, and VS Code MCP files

Project values beat user values and imported values. `disabledServers` beats `enabledServers`, which beats a server's own `enabled: false` value.

A project MCP file is executable configuration. Review every stdio command, argument, environment value, and remote URL before opening an untrusted repository with credentials available.

## Install or repair MCP support

Current Local Operator releases include the MCP SDK. For an older or intentionally minimal pip/source install, upgrade with the same Python environment that owns the `local-operator` executable:

```bash
python -m pip install --upgrade "local-operator[mcp]"
```

The desktop app may run a private virtual environment, so a random system `pip` can update the wrong copy. Prefer the app's backend updater. If diagnosing manually, first locate the running executable or Python environment; do not assume it is the global pip install.

## Troubleshooting

1. Run `local-operator mcp list` and confirm the effective target.
2. Inspect the winning config file and precedence.
3. Run the stdio command directly or check that the remote URL is reachable.
4. Restart the Local Operator session and read its MCP startup warning.
5. If only one server fails, fix that server. If every server reports the SDK missing, repair the Local Operator installation instead.
