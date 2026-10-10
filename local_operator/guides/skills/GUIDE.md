---
name: skills
description: "How Local Operator skills work: the builtin catalog, where skills come from and which root wins, how a skill reaches a session, and how to override or add one for a user."
---

# Skills

A skill is a folder with a `SKILL.md` whose description routes tasks to it. Skills come from your own roots, from other agent tools, and from the builtin catalog Local Operator ships with every release. This guide covers all three.

## What ships in the box

Fourteen builtin skills are packaged inside Local Operator itself. They are release content: every update refreshes them as part of the install, which is why you customise one by copying it out rather than editing the packaged files.

- `verification-before-completion` — prove work works before claiming it does: exercise the real path, read the actual output, say what remains unverified.
- `test-driven-development` — tests first: RED-GREEN-REFACTOR, and reproduce a bug as a failing test before fixing it.
- `systematic-debugging` — root-cause defects instead of guessing: reproduce, isolate, instrument boundaries, kill hypotheses.
- `engineering-principles` — smallest change, subtract before adding, data structures before logic, idempotency.
- `code-review` — diff-first review with severity classification and terminal verdicts.
- `responding-to-review` — answer findings one at a time: verify, fix/reject/defer with evidence, one batched round.
- `design-qa` — deterministic UI/UX checks (contrast, spacing, overlap, clipping, copy) plus judgement rules for the rest.
- `frontend-design` — deliberate interface design: plan before pixels, avoid the generic default look, hold a quality floor.
- `browser-qa` — exercise a web app through a real browser: recon, selectors, console errors, screenshot evidence.
- `data-analysis` — profile before trust: schema, keys, missingness, reproducible queries, provenance.
- `financial-analysis` — periods, units and currencies, reconciliation, magnitude sanity checks.
- `scheduling` — time zones done right: IANA names, DST edges, confirm in the attendee's own zone.
- `skill-authoring` — writing and improving skills: description as routing signal, progressive disclosure, budgets.
- `writing` — strip AI tells, plain language, active voice, reader-first structure.

## Where skills come from and who wins

Roots are scanned in order, and the first root holding a given name wins. Every shadowed loser is reported as a `Skill name conflict … (earlier root wins)` warning naming both files.

| Order | Root | Owner |
| --- | --- | --- |
| 1 | `<project>/.local-operator/skills` (walked up from the working directory, deepest first) | the repository |
| 2 | `~/.local-operator/skills` | you |
| 3 | Ecosystem roots: `~/.omp/agent/skills`, `~/.claude/skills`, `~/.codex/skills`, `~/.agents/skills` | other agent tools |
| 4 | The packaged builtin catalog | Local Operator (release content) |

Two environment knobs change the scan:

- `LOCAL_OPERATOR_SKILL_EXTRA_ROOTS` replaces the ecosystem set (colon-separated absolute paths; an empty value disables ecosystem scanning entirely). The builtin catalog is not part of that set, so it stays under every value.
- `LOCAL_OPERATOR_SKILL_MAX_DEPTH` (default 3, clamped to 1–5) caps how many directory levels below a root a skill may sit, so grouped libraries are found. The builtin catalog is flat, so it is discovered under every legal value.

## How a skill reaches a session

- The **description** is the routing signal. It is the only text sent before the skill is read, so it decides which tasks select the skill; write it with the vocabulary the task will use.
- Selection is semantic and **frozen per session** for prompt-cache stability: a new skill, a rename, or an edited description joins selection at the next session start. Edits to a body take effect on the next read.
- `$` in the composer completes skill names, `/skills` lists what the current folder resolves, and `skill://<name>` reads the body. `skill://<name>/references/<file>.md` reads one reference; the bare `skill://<name>` read also lists a skill's reference files.
- A skill that fails to load explains itself: read its `skill://` URL and the error names the cause (no `SKILL.md`, a blank description, malformed frontmatter, `enabled: false`, a frontmatter `name` that disagrees with the directory, or an earlier root shadowing it) and points at the fix.

## Customise or override a builtin

- **Override without editing it**: copy the skill to `~/.local-operator/skills/<name>/` (or a project root) keeping the same name. Your copy wins; the builtin is shadowed and reported once as a name conflict at session start. Rename your copy and both load, with no warning.
- **Silence one**: a same-named copy with `enabled: false` replaces the builtin with nothing; `hide: true` keeps direct `skill://` reads working but removes it from semantic selection.
- **Do not edit the packaged copy** — the next update replaces it.
- The conflict warning is not an error: it names the shadowed file, the winner, and the rule. It is exactly what shadowing a builtin looks like.

## Author and improve skills

The `skill://skill-authoring` skill is the playbook: description as routing signal, progressive disclosure, token budgets, and testing before shipping. Format mechanics and the extension surfaces are in `guide://extensions`.

## Place a skill for a user

Write `<root>/<name>/SKILL.md` with frontmatter `name` and `description`, and add `references/*.md` for material only some tasks need. A skill you create is readable immediately — the next `skill://` miss rescans the roots, so it resolves in this session and in subagents already running, with no restart. What waits for the next session is semantic selection: the skill is not auto-suggested until then, so tell the agent to `read skill://<name>` by name. Validate by reading it back: `skill://<name>` must return the body, not an error.
