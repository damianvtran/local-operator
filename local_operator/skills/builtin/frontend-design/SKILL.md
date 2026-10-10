---
name: frontend-design
description: "Designing interfaces with a deliberate point of view: plan before pixels, avoid the generic-default look, typography and colour discipline, a quality floor that is not negotiable. Use when creating or improving web UI."
---

# Frontend design

A deliberate point of view, then construction, then adversarial self-review. Generic is a decision too, and usually the wrong one.

## Pass 1: plan before pixels

Write down, before building:

- The audience and the ONE job of the screen.
- The primary action, and what the screen has to carry (real strings, real data volumes).
- The states: loading, empty, error, populated.
- Type plan: one scale (4 to 6 sizes, 2 to 3 weights). Colour plan: one accent, a neutral base, semantic roles (surface, border, text, danger). Say why each choice fits the audience.

Then attack the plan: describe what the default or template version of this screen would look like (hero, gradient, uniform card grid?). Remove or justify every piece of it that crept in. If the plan would work for any product, it is not a plan.

## Pass 2: build and self-critique

- Build the states as designed. The empty state is part of the design, not a fallback.
- Capture frames at narrow, standard, and wide (harness `browser` tool where available) and critique them against the plan: hierarchy, alignment, rhythm, legibility.
- Fix and recapture until the frames match the plan. Then look once more for what you stopped seeing.

## Anti-generic checklist

| Tell | Fix |
|---|---|
| Everything is a card in a uniform grid | Vary weight: some content is prose, some is data; not all containers are equal |
| Default gradient hero, centred everything | Choose one focal element; align to a grid on purpose |
| Three icon-title-blurb columns | Use the real content shape, or one strong statement with supporting detail |
| Filler copy ("empower your workflow") | Say what the control does, in the user's vocabulary |
| One font size everywhere | A scale, used consistently |
| Colour from a palette dump | One accent; colour for meaning, not decoration |
| Decoration that carries no meaning | Delete it; whitespace is a design element |
| Stock imagery that says nothing | Product truth, or nothing |

## Quality floor (not negotiable)

Ships without these or it does not ship:

- Responsive from the narrowest supported width.
- Focus-visible on every interactive element; keyboard order matches the visual order.
- Contrast: 4.5:1 normal text, 3:1 large text and meaningful UI borders.
- Reduced motion respected; motion earns its place.
- Real loading, empty, error, and populated states; no layout jump on load.
- Copy written for the user, no placeholders, no internal vocabulary. Checks: `skill://design-qa`.

## Red flags

- Designing to a single screenshot; the empty and error screens are left for "later".
- "It looks clean" with no plan, no narrow-width frame, and no state coverage.
- Importing a trend wholesale (glass effects, neon gradients) with no link to the brief.
- Polishing visuals while the quality floor is unmet.
