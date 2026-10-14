---
name: skill-authoring
description: "Writing and improving agent skills: description as routing signal, progressive disclosure, token budgets, testing before shipping, and improving from repeated mistakes. Use when authoring, editing, or reviewing a skill."
---

# Skill authoring

A skill is instructions that route: the description selects it, the body runs it. Write for both, and test before shipping.

## Structure

- `SKILL.md` frontmatter: `name` (kebab-case, equal to the directory name) and `description`. Flags and loading rules: `guide://extensions`.
- Body: the procedure, imperative, in the fewest lines that stay complete. Lead with the steps; put the why in one clause where a step would look arbitrary without it.
- References (`references/*.md`, at most 3): deeper material only some tasks need. The body must stand alone without them. Template: `references/template.md`.
- Budgets: description 220 characters or less; body 120 lines and about 6 KB or less. Every session lists the description, so cut hard.

## The description is the routing signal

- It is the only text seen before a skill is read. It must contain the task vocabulary a user would actually type.
- Third person, one or two sentences: what the skill does, then "Use when ..." for the trigger.
- Test it: write three prompts that should select the skill and two that should not. If the description cannot separate them, rewrite it.

## Body rules

- Imperative voice; checklists and tables over prose.
- Add a red-flags section wherever the failure mode is an agent rationalising a shortcut; those are the costly failures.
- State constraints where they matter ("never install a browser engine", budgets, portability rules).
- Do not restate what the model already knows, and do not duplicate agent role instructions; skills are shared knowledge, roles are assignments.

## Test before shipping

- Dry-run: hand the skill and a realistic task to a fresh agent context; watch whether the procedure is followable as written. Ambiguity shows up as invented steps.
- Pilot prompts: selection fires on the should-list and stays quiet on the should-not-list.
- Budget check: measure description characters, body lines, and bytes mechanically after every edit.

## Improvement loop

Fix a repeated mistake at the HIGHEST enforceable level: lint, then test, then prose. A rule in tooling cannot be forgotten; a rule in prose can.

| Level | Example | Use when |
|---|---|---|
| Tooling or lint | A checker that fails on the pattern | The rule is mechanical |
| Test | A test encoding the expected behaviour | The rule is behavioural |
| Skill prose | A red-flag line in the skill | The rule needs judgement |

Keep a rule-to-enforcement table in the skill when several rules exist; anything enforced below its level is a known gap, not a fix.

## Where skills live

- Roots, discovery order, and precedence: `guide://skills`.
- Format mechanics: `guide://extensions`. Template to copy: `references/template.md`.
