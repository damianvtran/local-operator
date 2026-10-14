# SKILL.md template

Copy, fill, and delete every comment. Format mechanics live in `guide://extensions`; roots and discovery in `guide://skills`.

```markdown
---
name: my-skill
description: <what it does, in the words a user would type>. Use when <trigger condition>.
---

# My skill

<One sentence: what this procedure guarantees.>

## Procedure

1. <First step, imperative.>
2. <Second step.>
3. <Third step.>

## Red flags

- <The shortcut that looks reasonable and is not.>

## References

- `references/details.md` for <the branch of the task that needs it>.
```

## Field notes

- `name`: kebab-case, equal to the directory name, stable. A rename breaks `skill://` links.
- `description`: third person, 220 characters or less. Test it with three should-select prompts and two should-not prompts.
- Body: keep to 120 lines or less; anything longer moves to `references/`.

## Checklist before shipping

- [ ] Frontmatter parses; name matches the directory.
- [ ] Description is a routing signal, not a summary of the body.
- [ ] Body leads with the procedure; red flags present where shortcuts tempt.
- [ ] Budgets measured: description characters, body lines, body bytes.
- [ ] Every reference file read and correct; the body works without them.
- [ ] Dry-run by an agent that has not seen the task; no invented steps.
