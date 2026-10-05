# ask-tui fleet score — round 2 (r3)

Every frame here was re-rendered from the round-2 remediation head (\`feat/all-asks-tui\` @ the push that
carries \`test(asks)\`/round-2 commits), from ONE process cwd, and differs from its \`pr-ask-tui-fleet-r2\`
twin **only** where the round-2 rulings changed pixels. The frames NOT listed here are **byte-identical**
to r2 — verified with \`cmp\` on the SVG pair, not asserted — and \`scripts/ask_shot.py\` +
\`scripts/approval_shot.py\` are byte-identical too (the ask-long-descriptions invariant holds).

| frame | what changed | why |
|---|---|---|
| \`sidebar-marks-100x30-{dark,light}\` | the \`?\` marker's ink | D12: the sidebar's mark now paints the derived \`chip-live\`; the frame's cursor row is the focused ground nobody solved for |
| \`sidebar-marks-80x24-dark\` | the footer | U13: the 29-cell floor now teaches \`ctrl+f\` beside the note |
| \`sidebar-gate-100x30-dark\` | the footer | U13 as above; \`!\` still keeps the cell while the count holds that session's ask |
| \`picker-asks-100x30-{dark,light}\` | the note's ink | D14/U12: the door paints \`muted\` (\`#565147\` light, 5.18:1) while the inert \`3 sessions\` legend keeps \`dim\` (\`#837c6d\`, 2.72:1) — plus the hover affordance, which a still cannot show |
| \`list-100x30-nocolor\` | **now a real colour-less render** | D11: the harness pops \`NO_COLOR\` by design, so the r2 "NO_COLOR frame" was a byte-identical copy of the colour one. The mode re-asserts it above the app import; md5 \`ecd382615de0…\` (the value the design round derived independently) and it is NOT equal to \`list-100x30-dark\` (\`10c8f6f57a96…\`) |
| \`list-130x30-dark\` | new (context) | the list at the width where the header's clause rides beside the segments |

**Numbers beside the stills.** Light marker ink, read out of the frame's own style block: \`#211e18\`
(= \`chip-live\`, not the raw \`#177b45\`) — **13.43:1** on the row ground (\`tint-select\` \`#dfeadf\`) and
**12.39:1** on the focused cursor ground (\`tint-select-hi\` \`#d2e3d2\`, where raw \`accent\` measured
**3.96:1**). Across the 54 registered ramps \`chip-live\` clears 4.0 on all four sidebar grounds; worst
4.66 (\`everforest\` on \`tint-select-hi\`). Picker note: \`#565147\` on \`overlay\` = 5.18:1 light / 6.51:1
dark, against the legend's unchanged 2.72 / 3.43.

**Geometry.** Unchanged surfaces, re-measured on these frames: the list header is exactly one painted
row; each ask spends exactly one painted row; no \`AskQueueList\`/\`AskBar\`/\`SessionSidebar\` scrollbar
appeared (\`scrollbar [False, False]\`); the three filter segments fit at 100 columns with the clause
yielding first. Sidebar footers, read from the painted bytes: \`esc return · ctrl+f · asks: 4\` at both
80×24 and 100×30 — the chord is taught at every width the panel renders, which it was not before.
