# PR #4238 round-1 remediation frames (head 0d9fb8b7)

Real `OperatorApp` (stylesheet loaded) via `run_test` -> `save_capture` SVG ->
`rsvg-convert`, isolated HOME/config (`scripts.visual_capture.isolate_capture`).
Real resolver over a seeded config: display store, reservation (pending),
journal rows, a real Radient OAuth row. Doubles: the hub wire (a table of rows
by seq, never the classification fetch) and `Store.hold_states_readonly`
(review/transient hold rows). `before-*` = PR head df870ed1, same rig.

- `*-collapsed-{dark,light}`: 8 fresh cards, never opened (D1/D2/D3).
- `*-states-open-*`: admitted (inline code, D6), held, transient (D3), refused
  (hub) open; state lines in header ink (D5).
- `*-states-reclosed-*`: same cards closed again (refused header now known).
- `*-long-{110,60}-*`: 30-line body clamped at 12 rows; `-focused` shows `· m`
  (D4). Long card: 17 rows clamped at both widths, show-more row index 16.
