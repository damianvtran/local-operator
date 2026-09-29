#!/bin/sh
# PostToolUse(Bash): after a non-draft `gh pr create` or `gh stack submit`, tell the
# main agent to run /fix-pr-reviews in the foreground.
input=$(cat)
# ponytail: subagents carry agent_id; only the main session should drive the loop
[ -n "$(printf '%s' "$input" | jq -r '.agent_id // empty')" ] && exit 0
cmd=$(printf '%s' "$input" | jq -r '.tool_input.command // empty')
case "$cmd" in
  *"gh pr create"*) case "$cmd" in *--draft*|*" -d "*) exit 0 ;; esac ;;
  *"gh stack submit"*) case "$cmd" in *--draft*) exit 0 ;; esac ;;
  *) exit 0 ;;
esac
jq -n '{hookSpecificOutput: {hookEventName: "PostToolUse", additionalContext:
  "A PR was just opened. Invoke the fix-pr-reviews skill on it now, in this main session (not a background agent or subagent), and drive it to approval. For a gh stack submit, run it over every PR in the stack."}}'
