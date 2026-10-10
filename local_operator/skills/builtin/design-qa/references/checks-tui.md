# Terminal UI checks

Run against the real terminal app (the build that loads the shipped styles), not a bare test host. Record the terminal size and paste actual text in every finding.

## Resize

- Resize through the supported range: smallest promised width, standard, very wide, very short. Watch for wrapped borders, duplicated headers, content jumping, and lost scrollback.
- Capture before and after frames per size. The first frame after a resize is not settled until the redraw completes; wait for it.

## Width and encoding

- Long lines: do they truncate with an affordance, wrap, or overflow the pane? Where does it break?
- Double-width glyphs (CJK, emoji) at the pane edge: check for a one-column gap or a broken border.
- Test the narrowest width the product promises (for example 40, 50, 60 columns) and one below it to see the failure mode.

## Input and latency

- Every action reachable by keyboard; tab and arrow order follows the visual order.
- Latency on the heaviest screen: time from keypress to visible change. Over 100ms starts to feel laggy; over 200ms is a defect worth reporting.
- Key repeat, paste of multi-line text, and terminal-resize during a prompt.

## Colour and themes

- Run every supported theme, including a no-colour terminal: text stays readable, semantic colours keep their meaning.
- Never colour-only encoding: state changes carry a glyph or label too, so they survive colour-blindness and monochrome.
- NO_COLOR and dumb-terminal fallback: layout must not collapse.

## States

- Loading: shown, or is the pane blank? Blank is a finding.
- Empty: does it say what to do next?
- Error: visible, actionable, not swallowed by an alternate screen.
- Dialogs and prompts: cancel path works; destructive confirmations exist and read clearly.

## Evidence format

Per check: terminal size (cols x rows), the command, the actual output text, and the exported frame path where the app can produce one. Example: `120x40 | resize to 60 cols: header wraps to two lines, border breaks | frames/tui-60.svg`.
