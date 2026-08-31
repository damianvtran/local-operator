# Team/agent picker catch-up and highlight evidence

Captured with the real `OperatorApp` and shipped stylesheet via Textual
`run_test`, using `env -u NO_COLOR TERM=xterm-256color`. `before-*.svg` comes
from the original base (`85b0737a`); `after-*.svg` comes from this branch at
100×30 except the explicitly narrow 40×16 frames. SVGs were rendered with Quick
Look and viewed; `after-montage.png` stacks every remediation state.

## Frames

1. `after-agent-pending`: `/agent aud` while session construction is blocked.
   The restrained, non-selectable `loading agent roster…` row reserves one
   picker row. Pending Tab/Enter are interaction-tested no-ops: exact buffer
   and caret survive until real rows arrive.
2. `after-agent-filled`: the same unchanged `/agent aud` after the delayed
   registry arrives; the highlighted `auditor` row replaces the reserve in
   place.
3. `after-agent-settled`: consecutive post-adoption frame; identical geometry
   proves there is no settling reflow. `after-agent-r2-montage.png` stacks all
   three frames in that order; it was rendered via Quick Look and viewed.
4. `after-catchup-empty`: `/team lop` while session construction is blocked.
   The restrained, non-selectable `loading teams…` row reserves one picker row.
5. `after-catchup-filled`: same unchanged buffer after the delayed `lopdev`
   registry arrives; the real row replaces the reserve in place.
6. `after-catchup-filled-settled`: consecutive post-adoption frame; identical
   geometry proves there is no settling reflow. These existing D1 frames were
   recaptured on the final implementation and remain geometry-identical.
7. `after-team-tab-parked`: interaction evidence, not an inferred still: the
   capture script starts at bare `/team `, presses Tab, asserts the buffer is
   `/team lopdev ` (not `chart`), and captures the switch/send hint.
8. `after-team-chart-reserved`: `/team chart lopdev` preserves the reserved
   `chart` treatment while the compound picker row remains selectable.
9. `after-narrow-overflow`: 40×16 with twelve team matches, truncated columns,
   four visible rows and the `… 7 more` overflow marker.
10. `after-narrow-parked`: 40×16 parked switch/send state remains legible without
   a picker hole or scrollbar.

The earlier `typed-*` and `inline-multiline` frames remain as evidence for
hand-typed and multiline exact-name highlighting.

## Geometry

The real-Session catch-up capture asserted these exact values for BOTH the
`/agent aud` pending/first-filled/settled triplet and the existing `/team lop`
triplet:

```text
composer y=24 · picker y=25 height=1 · status y=26
welcome y=1 · version SVG y=337.2
screen=98×28 · virtual=98×28 · scrollbar=False
```

All three `/agent` metric dictionaries are exactly equal, as are all three
`/team` dictionaries. The pending and populated rows render at the same SVG
coordinate (`y=630`); neither transition creates a scrollbar.

Before D1 remediation, the same capture measured composer y=25 while empty and
y=24 after the row arrived (a one-cell / 24.4px jump); the original SVG review
likewise measured the version line moving from y=361.6 to y=337.2. After, the
pending notice and first real row both render at SVG y=630, so adoption replaces
content without changing any key coordinate.

At 40×16 the overflow frame reports `screen=38×14`, `virtual=38×14`, and no
scrollbar; the parked frame also keeps virtual equal to screen.
