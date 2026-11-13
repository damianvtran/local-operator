---
name: dangerous-delete
scope: tool:bash
resolver: destructive-paths
action: review
question: Did the user's request clearly call for deleting, moving away or overwriting exactly these paths, or did this session or its subagents create them (evidence id or ledger fact) with no unfinished uncommitted work in them?
---
Deleting, moving away or overwriting the harness state directory, its config or
secrets, the home directory, another session's files or a repository root needs
the user's explicit request — unless this session or its subagents created it
(an evidence id or a ledger fact proves it) and it holds no unfinished
uncommitted work: removing your own finished worktree or scratch needs no ask.
Target the narrowest path you created, never build a delete path from variables
or substitutions, and ask the user only about paths you did not create.
