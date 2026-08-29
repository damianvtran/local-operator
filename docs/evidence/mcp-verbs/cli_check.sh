#!/bin/bash
# Prove `lop mcp list|add|remove` is UNCHANGED after the config.py refactor:
# identical stderr text and exit codes. Temp HOME, so the operator's real
# ~/.local-operator/mcp.json is never read or written.
# Usage: cli_check.sh <repo-root>
set -u
REPO="${1:-/tmp/lop-mcp}"
H=$(mktemp -d /tmp/mcp-cli-home.XXXX); W=$(mktemp -d /tmp/mcp-cli-cwd.XXXX)
export HOME="$H"; cd "$W" || exit 1
run() { echo "--- \$ lop mcp $*"; PYTHONPATH="$REPO" "$REPO/.venv/bin/python" -m local_operator.cli mcp "$@" 2>&1; echo "    exit=$?"; }
run list
run add demo-stdio --command npx
run add demo-http --url https://demo.example/mcp
run add demo-oauth --url https://oauth.example/mcp --oauth
run list
echo "--- error paths (stderr text + exit code must be unchanged)"
run add demo-stdio --command npx                        # duplicate
run add both --command npx --url https://x.example/mcp  # both
run add neither                                         # neither
run add 'bad name!' --command npx                       # invalid name
run add stdio-oauth --command npx --oauth               # oauth on stdio
run remove nope                                         # not found
run remove demo-http
run list
echo "--- resulting global config"
cat "$HOME/.local-operator/mcp.json"
rm -rf "$H" "$W"
