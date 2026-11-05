---
name: dangerous-delete
scope: tool:bash
resolver: destructive-paths
action: review
question: Did the user's request clearly call for deleting, moving away or overwriting exactly these paths?
---
Deleting, moving away or overwriting the harness state directory, its config or
secrets, the home directory, another session's files or a repository root needs
the user's explicit request. Target the narrowest path you created (your
scratchpad), never build a delete path from variables or substitutions, and ask
the user when unsure.
