# Web checks

Run against the rendered page with the harness `browser` tool. Cite the actual value in every finding.

## 1. Console and network (run first)

- Open, await load, then read the logs: errors and warnings are defects until explained.
- Note failed requests (4xx, 5xx, blocked assets) and what triggered them.

## 2. Contrast

- Floors: 4.5:1 for normal text; 3:1 for large text (roughly 24px, or 18.7px bold) and for meaningful UI borders and icons.
- Read the computed text colour and the effective background (walk ancestors for the painted background; account for opacity, overlays, and gradients). Compute the ratio; two colours that look distinct on a good monitor can still fail.
- Check states with their own colours: hover, focus, disabled, placeholder, and the dark theme.

## 3. Spacing, overlap, clipping

- Read bounding boxes and compute the gaps; never eyeball padding.
- Clipping: for every scrollable or truncated region, compare content size to viewport size (scrollWidth vs clientWidth, scrollHeight vs clientHeight). Content larger than its box that has no affordance is cut off.
- Overlap: compare bounding boxes of elements that must not intersect (bars, toasts, floating buttons, sticky headers).

## 4. Target sizes

- Pointer targets: 44x44 px minimum (24x24 acceptable only in dense desktop UI with generous spacing); read the box, not the visual.
- Check spacing between adjacent targets (mis-taps come from gaps, not just sizes).

## 5. Focus and keyboard

- Tab through the primary flow: focus visibly at every step, in an order that matches the layout.
- Focus must not be trapped by, or lost on close of, dialogs and menus.

## 6. Responsive

- Narrow to the smallest supported width: check reflow, unintended horizontal scroll, truncation, and tap targets.
- Zoom to 200%; the layout must survive without overlap or lost content.

## 7. Motion

- With reduced motion requested, animations stop or reduce.
- Capture consecutive frames around any transition; the settled state is the screenshot that counts.

## 8. Assets

- Images: meaningful ones have alt text; decorative ones have empty alt. No broken or stretched assets.
- Fonts: the intended family actually loaded (a fallback font is a silent failure). Check headings and body separately.

## Evidence format

Per check: the action, the actual value or reading, and the frame path. Example: `contrast .muted-on-card: 3.1:1 (needs 4.5) | frames/card-dark.png`.
