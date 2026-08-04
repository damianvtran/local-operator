You are summarizing the conversation so far so the agent can continue with a
compacted context. Produce a structured summary in EXACTLY the sections below.
Be faithful and specific: the summary replaces the original messages, so
anything worth remembering must survive here.

Rules:
- Preserve the user's UNANSWERED QUESTIONS verbatim, word for word.
- Keep exact file paths, symbol names, identifiers, commands, and error
  messages verbatim. Never paraphrase names or numbers.
- Keep decisions with their rationale, not just outcomes.
- Do not invent anything that is not in the transcript. If a section has no
  content, write "None." — keep the section anyway.
- Write in plain markdown; no preamble, no commentary about the summary itself.

## Goal

What the user wants, in one or two sentences, in the user's terms.

## Constraints

Explicit limits, requirements, and preferences (platforms, style rules,
deadlines, "do not touch X", approval policies).

## Progress

What has been done so far, in order, with results: files changed, commands
run, outcomes, tests passed or failed.

## Key Decisions

Decisions made during the conversation and why each was chosen, especially
where alternatives were rejected.

## Next Steps

The immediate next actions the agent should take, in order.

## Critical Context

Anything that does not fit above but would change behavior if lost: open
questions asked of the user, environment quirks, credentials/config locations,
exact error text encountered, pending side effects.

{{#if previous_summary}}
## Previous summary (fold into the sections above, then drop this section)

{{previous_summary}}
{{/if}}

## Conversation to summarize

{{transcript}}
{{#if files}}

<files>
{{files}}
</files>
{{/if}}
