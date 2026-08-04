# Scout report: oh-my-pi TUI + MCP subsystems

> **Delivery note:** this was meant to land at `local://scout-omp-tui-mcp.md`, but this agent has no write tool and `node_repl` is sandboxed read-only (EPERM on cwd, tmpdir, and the `local/` dir alike). The full report is inlined here; copy it to that path if a file is required.

Read-only recon of `~/oss/oh-my-pi` for the local-operator rewrite. Two packages matter:

- `packages/tui` — standalone terminal rendering engine (`@oh-my-pi/pi-tui`). No React / ink / blessed / curses. **Raw ANSI, hand-rolled differential renderer.**
- `packages/coding-agent` — the agent. `src/modes/` is the TUI *integration* layer; `src/mcp/` is a **from-scratch MCP client** (no official SDK dependency).

Authoritative prose docs (read before implementing): `docs/tui.md`, `docs/tui-core-renderer.md`, `docs/tui-runtime-internals.md`, `docs/theme.md`, `docs/keybindings.md`, `docs/mcp-config.md`, `docs/mcp-runtime-lifecycle.md`, `docs/mcp-protocol-transports.md`.

---

## 1. TUI architecture

### 1.1 Primitives — what it is built on

**Nothing.** `packages/tui/src/tui.ts` (4273 lines) writes ANSI escape bytes directly to a `Terminal` abstraction (`packages/tui/src/terminal.ts` -> `ProcessTerminal`). Constants at the top of `tui.ts`:

```ts
const HIDE_CURSOR        = "\x1b[?25l";
const SYNC_OUTPUT_BEGIN  = "\x1b[?2026h";   // DEC 2026 synchronized output
const SYNC_OUTPUT_END    = "\x1b[?2026l";
const DISABLE_AUTOWRAP   = "\x1b[?7l";
const ENABLE_AUTOWRAP    = "\x1b[?7h";
const LINE_TERMINATOR    = "\x1b[0m\x1b]8;;\x07"; // close SGR + OSC-8 hyperlink per row
const ERASE_LINE         = "\x1b[2K";
const MOUSE_TRACKING_ON  = "\x1b[?1000h\x1b[?1003h\x1b[?1006h";
const ALT_SCREEN_ENTER   = "\x1b[?1049h";
const PAINT_BEGIN = HIDE_CURSOR + SYNC_OUTPUT_BEGIN + DISABLE_AUTOWRAP;
const PAINT_END   = ENABLE_AUTOWRAP + SYNC_OUTPUT_END;
```

Most important architectural decision: the app runs on the **normal screen**, not the alt screen. Alt screen is reserved for specific fullscreen overlays. This preserves native scrollback, native mouse selection, and a transcript that survives process exit.

### 1.2 The Component contract

```ts
export interface Component {
  render(width: number): readonly string[];       // returns PHYSICAL ROWS
  handleInput?(data: string): void;
  wantsKeyRelease?: boolean;                       // Kitty key-release events
  invalidate?(): void;                             // theme change / force re-render
  setIgnoreTight?(ignore: boolean): any;
  dispose?(): void;                                // idempotent teardown
}

export interface Focusable {
  focused: boolean;
  setUseTerminalCursor?(useTerminalCursor: boolean): void;
}

export const CURSOR_MARKER = "\x1b_pi:c\x07";      // focused component embeds this;
                                                   // TUI extracts it + parks the hw cursor
export function isFocusable(c: Component | null): c is Component & Focusable;
```

**Critical invariant — the whole memoization scheme rests on it:** the returned array is component-owned and immutable to the caller. An unchanged component MUST return the *same array reference*; a changed one MUST return a *fresh* array. Reference equality is the engine's proof that rows are byte-identical.

`Container` (`tui.ts` ~486) is the composite: `children: Component[]`, `addChild` / `removeChild` / `clear` / `disposeChildren`, and a `render()` memoizing the concatenation keyed on (width, child array refs). `TUI extends Container` — the root is itself a container.

### 1.3 The render loop

Not a fixed-tick loop. **Demand-driven, throttled, adaptive.**

```ts
tui.requestRender(force = false, options?: { clearScrollback?: boolean }): void
tui.requestComponentRender(component: Component): void   // component-scoped partial compose
```

Scheduling (`#scheduleRender`, `tui.ts` ~2354):

- `#MIN_RENDER_INTERVAL_MS = 1000/30` -> 30 fps cadence ceiling.
- **Adaptive backpressure**: `#lastFrameCostMs` is measured per frame; the next frame starts no sooner than `last_frame_start + 2 x last_frame_cost` (~50% duty cycle), floored around 5 fps. Prevents CPU busy-loop on heavy transcripts (issue #4145).
- `#INPUT_RENDER_GRACE_MS` delays a frame briefly after a keystroke so bursts coalesce.
- Multiplexer resize debounce `#MULTIPLEXER_RESIZE_DEBOUNCE_MS = 50` (tmux/zellij repaint mid-reflow otherwise, issue #2088).
- Resize viewport fast path: during a drag, `#renderResizeViewport` paints only the visible tail via `ViewportTailProvider.renderViewportTail(width, maxRows)`; an authoritative full paint replays on settle.
- ConPTY post-full-paint settle window (~150 ms) absorbs spinner-driven renders that would desync Windows Terminal's viewport tracker (issue #2095).
- Everything is injectable: `RenderScheduler { now, scheduleImmediate, scheduleRender }` via `TUIOptions.renderScheduler` — the engine is deterministically testable.

Frame pipeline `#doRender()` (`tui.ts` ~2811):

1. **Compose** — walk root children, call `render(width)`, collect `getNativeScrollbackLiveRegionStart()` seams, build `FrameSegment[]`.
2. **Audit committed prefix** — `findCommittedPrefixResync(frame, prefix, verifiedTo, finalTo)` (exported, ~862) samples up to `RESYNC_TAIL_SAMPLES = 8` non-blank rows within the last `RESYNC_TAIL_LOOKBACK = 24`, SGR-stripped.
3. **Classify intent** — `type RenderIntent = { kind: "fullPaint"; clearScrollback } | { kind: "update"; chunkTo; windowTop }`.
4. **Extract cursor markers, prepare width-safe lines, slice window, composite overlays** (overlays only into the screen-coordinate window).
5. **Emit** via `#emitFullPaint` or `#emitUpdate` (scroll-append / in-window diff / seam rewrite).

### 1.4 The append-only scrollback ledger (the core idea)

The engine **cannot observe the terminal's scroll position** (no POSIX API; ConPTY's probe lies). So it never guesses. It maintains:

- **C = `#committedRows`** — frame rows `[0, C)` have entered native scrollback and are **immutable**.
- **W = `windowTopRow`** — frame row mapped to grid row 0; visible window `[W, W+height)`.
- **B = live-region boundary** — first row that may still mutate.

Ordinary frame: `W = max(C, L - height)`, new commit end `= max(C, W)`. Only the chunk between old and new commit indices is written into history.

Components declare the seam:

```ts
export interface NativeScrollbackLiveRegion {
  getNativeScrollbackLiveRegionStart(): number | undefined;
  isNativeScrollbackLiveRegionPinned?(): boolean;   // dashboards: clip instead of snapshot
}
export interface NativeScrollbackCommittedRows { setNativeScrollbackCommittedRows(rows: number): void; }
export interface NativeScrollbackReplay { prepareNativeScrollbackReplay(): void; }
export interface RenderStablePrefix { getRenderStablePrefixRows(): number; }   // reading CONSUMES it
export interface ViewportTailProvider { renderViewportTail(width, maxRows): readonly string[]; }
export interface OverlayFocusOwner { ownsOverlayFocusTarget(c: Component): boolean; }
```

Hard rules (`docs/tui-core-renderer.md` section 3): never add a second `CSI 3 J` (ED3) callsite — ED3 flows only through `#emitFullPaint({ clearScrollback: true })` for explicit gestures, never inside multiplexers; ordinary emitters never rewrite a committed row; never probe the viewport or fork on platform in the update path; never throw in the render hot path (clamp with `truncateToWidth`); park the hardware cursor at real content bottom, not padded window bottom; cursor writes go inside the synchronized-output frame before ESU.

### 1.5 Screen composition (the actual layout)

`packages/coding-agent/src/modes/interactive-mode.ts` lines ~716-994. Root children of the `TUI` container, in order:

```
Spacer(1)
WelcomeComponent                       (startup banner, optional)
[changelog block]                      (optional)
chatContainer           : TranscriptContainer     <- the message stream
pendingMessagesContainer: AnchoredLiveContainer
todoContainer           : AnchoredLiveContainer   <- sticky todo HUD
subagentContainer       : AnchoredLiveContainer   <- in-flight subagent HUD
btwContainer            : AnchoredLiveContainer
omfgContainer           : AnchoredLiveContainer
errorBannerContainer    : AnchoredLiveContainer
modelCycleContainer     : AnchoredLiveContainer
statusContainer         : AnchoredLiveContainer   <- "Working..." loader / transient status
statusLine              : StatusLineComponent     <- ONLY renders hook statuses here
hookWidgetContainerAbove: Container
editorContainer         : Container -> CustomEditor
hookWidgetContainerBelow: Container
```

Then `this.ui.setFocus(this.editor)`.

**The status line is not a separate bottom row.** It is rendered *into the editor's top border*:

```ts
this.editor.setTopBorderProvider(availableWidth => this.statusLine.getTopBorder(availableWidth));
```

Zero extra rows for the status display. Worth stealing.

`AnchoredLiveContainer` (defined in `interactive-mode.ts` ~362) reports a seam at row 0 so its rebuilt-in-place rows never commit to scrollback — otherwise stale HUD duplicates pile up above the live copy.

Editor height: `computeEditorMaxHeight(terminalRows)` — comfortable band [6, 18], shrinking on small terminals but never below `EDITOR_MIN_RENDERED_ROWS = 3` while reserving `EDITOR_MIN_CHROME_ROWS = 4` for transcript + status.

### 1.6 Overlays and focus

```ts
tui.showOverlay(component, options?: OverlayOptions): OverlayHandle  // { hide, setHidden, isHidden }
tui.hideOverlay(): void
tui.hasOverlay(): boolean
tui.setFocus(component: Component | null): void
tui.getFocused(): Component | null
tui.addInputListener(l): () => void      // pre-dispatch interception, can consume/rewrite
tui.addStartListener(l): () => void
tui.setShowHardwareCursor(b) / getShowHardwareCursor()
tui.setMaxInlineImages(cap) / clearInlineImages()
tui.setScrollbackRebuild(b) / getScrollbackRebuild()
tui.resetDisplay()                        // Alt+L chord
```

`OverlayOptions` supports `SizeValue = number | percent-string`, `OverlayAnchor = "center" | "top-left" | ... | "bottom-center"`, `OverlayMargin`, and a `visible(cols, rows)` predicate. Overlays **freeze commits** while visible.

### 1.7 Width model — one model, everywhere

`packages/tui/src/utils.ts`:

```ts
export function visibleWidth(str: string): number
export function truncateToWidth(text, maxWidth, ellipsisKind?): string
export function sliceWithWidth(line, startCol, length, strict?): SliceResult
export function wrapTextWithAnsi(text: string, width: number): string[]
export function extractSegments(line, beforeEnd, afterStart, ...)
export function replaceTabs(text: string): string
export function padding(n: number): string
export function publishLineWidths(lines, widths) / getPublishedLineWidths(lines)
export function encodeTextSized(text, opts: TextSizingOptions)   // OSC 66 (kitty)
export function setHangulCompatibilityJamoWidth(w) / getWidthConfigEpoch()
```

Slicing/wrapping run on the Rust native engine (`@oh-my-pi/pi-natives`, `unicode-width`); `visibleWidth` uses `Bun.stringWidth` **pinned to the same model** (`countAnsiEscapeCodes: false, ambiguousIsNarrow: true`). UAX#11. *Mixing width models between measure and slice previously caused crashes* — the most repeated warning in the codebase. Tabs re-added at `DEFAULT_TAB_WIDTH`; OSC 66 sized spans re-added as `scale x width`.

### 1.8 Terminal capabilities

`packages/tui/src/terminal-capabilities.ts` resolves a `TERMINAL` profile once at import from `TERMINAL_ID` + env sniffing. Detection helpers are pure over (env, platform) and unit-testable:

- `shouldEnableSynchronizedOutputByDefault(env, id)` -> DEC 2026, reconciled at runtime by the DECRQM mode-2026 report.
- `detectRectangularSgrSupport(id, env)` -> DECCARA rectangular background fills, kitty only (`packages/tui/src/deccara.ts`, `planDeccaraFills`).
- `supportsScreenToScrollback` -> kitty ED22, used once on the initial paint to preserve the pre-existing shell screen.
- `ImageProtocol` (Kitty graphics / Sixel), `isInsideTerminalMultiplexer()`, `setCellDimensions`.

Probing lives in `terminal.ts` (`ProcessTerminal`): capability queries are fused with a bare DA1 (`CSI c`) sentinel; `#privateCsiResponseBuffer` reassembles split private-CSI replies (abandoning after 256 bytes or a new ESC so real keys still reach input); `#da1SentinelOwners` is a **typed FIFO** so a keyboard DA1 cannot be mistaken for an OSC-11 / DECRQM / graphics sentinel. Stdin escape reassembly: `stdin-buffer.ts`.

Env escape hatches: `PI_NO_SYNC_OUTPUT`, `PI_TUI_SYNC_OUTPUT`, `PI_FORCE_SYNC_OUTPUT`, `PI_NO_DECCARA`, `PI_NO_SGR_COALESCE`, `PI_NO_KITTY_PLACEHOLDERS`, `PI_TUI_RESIZE_IN_PLACE`.

### 1.9 Inline images

`kitty-graphics.ts` + `components/image.ts`. **Transmit-once, place-many.** `ImageBudget` keeps the most recent N images live (`DEFAULT_MAX_INLINE_IMAGES`, settable via `tui.setMaxInlineImages(cap)`); demotion deletes pixels by id and falls back to *height-preserving* text (reserved rows + fallback line) so committed rows below never shift. Never re-emit base64 per frame.

### 1.10 Component library

`packages/tui/src/components/`: `box.ts`, `cancellable-loader.ts`, `editor.ts`, `image.ts`, `input.ts`, `loader.ts`, `markdown.ts`, `scroll-view.ts`, `select-list.ts`, `settings-list.ts`, `spacer.ts`, `tab-bar.ts`, `text.ts`, `truncated-text.ts`. All re-exported from `packages/tui/src/index.ts`.

Coding-agent-side shared render helpers, `packages/coding-agent/src/tui/`:
- `output-block.ts` — `renderOutputBlock(options: OutputBlockOptions, theme): string[]`, `CachedOutputBlock`, `framedBlock(theme, build)`, `markFramedBlockComponent`, `outputBlockContentWidth(...)`. This is the boxed tool-result frame.
- `status-line.ts` — `renderStatusLine(options: StatusLineOptions, theme): string` (icon + title + meta, one row).
- `utils.ts` — `Hasher` (render-cache keys, bigint), `RenderCache { key: bigint; lines: string[] }`, `buildTreePrefix`, `getTreeBranch`, `getTreeContinuePrefix`, `padToWidth`, `getStateBgColor(state)`.
- `types.ts` — `type State = "pending"|"running"|"success"|"error"|"warning"`, `TreeContext { index, isLast, depth }`.
- `code-cell.ts`, `file-list.ts`, `tree-list.ts`, `hyperlink.ts`, `width-aware-text.ts`.

### 1.11 Test harness (copy the *idea*)

`packages/tui/test/render-stress-harness.ts` drives the renderer's **real emitted ANSI** into a ghostty-web `VirtualTerminal` across randomized op sequences and parameterized terminal shapes, validated against an independent **shadow commit ledger** (a reimplementation of the section 1.4 math fed only by observed frames and observed bytes). Per op it asserts the whole tape (scrollback + grid) equals `shadowTape + window slice`, row for row, including across resizes; that scrolled readers stay pinned; that multiplexer pane history grows by exactly the committed chunk. Plus `render-regressions.test.ts`, `streaming-scrollback-defer.test.ts`, and per-issue repro tests.

---

## 2. Theme system

Implementation: `packages/coding-agent/src/modes/theme/theme.ts` (~3100 lines). Schema mirror: `src/modes/theme/theme-schema.json`. Built-ins: `src/modes/theme/defaults/` JSON files plus `dark.json` / `light.json`, compiled into `defaultThemes`.

### Shape

```jsonc
{
  "name": "my-theme",
  "vars":   { "accent": "#7aa2f7", "muted": 244 },     // optional, recursive refs
  "colors": { /* ~56 REQUIRED tokens */ },
  "export": { "pageBg": "...", "cardBg": "...", "infoBg": "..." },  // optional (HTML export)
  "symbols": {
    "preset": "unicode" | "nerd" | "ascii",
    "overrides": { "<SymbolKey>": "..." },
    "spinnerFrames": string[] | { "status"?: string[], "activity"?: string[] }
  }
}
```

Color value forms: hex `"#RRGGBB"`, 256-index `0..255`, a `vars` reference string, or empty string = terminal default (SGR 39 / 49).

### Token groups (all required except `thinkingMax`)

| Group | Count | Tokens |
|---|---|---|
| Core text / borders | 11 | `accent border borderAccent borderMuted success error warning muted dim text thinkingText` |
| Background blocks | 7 | `selectedBg userMessageBg customMessageBg toolPendingBg toolSuccessBg toolErrorBg statusLineBg` |
| Message / tool text | 5 | `userMessageText customMessageText customMessageLabel toolTitle toolOutput` |
| Markdown | 10 | `mdHeading mdLink mdLinkUrl mdCode mdCodeBlock mdCodeBlockBorder mdQuote mdQuoteBorder mdHr mdListBullet` |
| Diff + syntax | 12 | `toolDiffAdded/Removed/Context`, `syntaxComment/Keyword/Function/Variable/String/Number/Type/Operator/Punctuation` |
| Mode / thinking borders | 8 (+1) | `thinkingOff/Minimal/Low/Medium/High/Xhigh`, optional `thinkingMax`, `bashMode pythonMode` |
| Status line segments | 13 | `statusLineSep/Model/Path/GitClean/GitDirty/Context/Spend/Staged/Dirty/Untracked/Output/Cost/Subagents` |

### Runtime API

```ts
export class Theme {
  fg(color: ThemeColor, text: string): string;   // ansi + text + SGR39
  bg(color: ThemeBg,    text: string): string;   // ansi + text + SGR49
  bold(t) / italic(t) / underline(t) / strikethrough(t);
  icon: SymbolTheme; boxRound; boxSharp; tree;
  getThinkingBorderColor(level); getBashModeBorderColor(); getPythonModeBorderColor();
  getMajorThemeColorHexes(); accentSurfaceLuminance;
}
export let theme: Theme;                                  // module-level singleton (live binding)
export function initTheme(settings)
export function setTheme(name, watch?): { success: boolean; error?: string }
export function previewTheme(name)                        // live preview, does not persist
export function setThemeInstance(t: Theme): void
export function onThemeChange(cb: (e: ThemeChangeEvent) => void): () => void
export function getThemeEpoch(): number                   // bump = invalidate every render cache
export function getAvailableThemes(): ThemeInfo[]
export function highlightCode(code, lang?, theme?): string[]     // native highlighter
export function enableAutoTheme(e?) / setAutoThemeMapping(mode, name) / onTerminalAppearanceChange(mode, e?)
export function startMacOSAppearanceReprobeFallback(terminal): () => void
export function stopThemeWatcher() / isLightTheme(name?)
// Adapters — how sub-libraries get themed without importing Theme:
export function getMarkdownTheme(): MarkdownTheme
export function getSelectListTheme(): SelectListTheme
export function getEditorTheme(): EditorTheme
export function getSettingsListTheme(): SettingsListTheme
export function getSymbolTheme(): SymbolTheme
```

**Re-theming flow:** `setTheme(name)` -> load/validate/resolve vars (throws on missing or circular refs) -> convert to ANSI per `detectColorMode()` (truecolor via `Bun.color(..., "ansi-16m")` vs `ansi-256`) -> replace the global `theme` -> fire `onThemeChange` -> app calls `tui.invalidate()` (propagates through every `Container` child, dropping cached rows and bumping the `TranscriptContainer` generation) -> `requestRender(true)`. On failure it falls back to built-in `dark`.

Color-mode detection: `COLORTERM=truecolor|24bit` -> truecolor; `WT_SESSION` -> truecolor; `TERM` in `dumb`/`linux`/empty -> 256color; else truecolor.

Auto dark/light slot: OSC 11 background luminance -> `COLORFGBG` (index < 8 = dark) -> macOS appearance fallback (only for the known-broken macOS/Zellij OSC-11 path) -> dark. Defaults `theme.dark = "titanium"`, `theme.light = "light"`, `symbolPreset = "unicode"`.

Custom themes dir: `~/.omp/agent/themes/` (or the `PI_CODING_AGENT_DIR` override). Built-ins take precedence on name collision and are effectively not watched. The watcher does a debounced reload; errors or a temporarily missing file keep the last good theme. `colorBlindMode` HSV-shifts only `toolDiffAdded` (green toward blue), and only when the resolved value is hex.

Persisted keys live in `~/.omp/agent/config.yml`: `theme.dark`, `theme.light`, `symbolPreset`, `colorBlindMode`.

Box drawing: all chrome uses `boxRound` tokens (rounded corners + edges) with junctions (tee/cross) sourced from `boxSharp` since rounded forms do not exist. Markdown tables are the only fully-sharp consumer.

Symbol presets `unicode | nerd | ascii`, precedence: settings override -> theme JSON `symbols.preset` -> `unicode`. Two spinner types: `status` (~12.5 fps, loaders and tool indicators) and `activity` (~30 fps, markdown progress).

---

## 3. Message rendering

### 3.1 TranscriptContainer

`packages/coding-agent/src/modes/components/transcript-container.ts` (528 lines).
`class TranscriptContainer extends Container implements NativeScrollbackLiveRegion, NativeScrollbackCommittedRows, RenderStablePrefix, ViewportTailProvider`.

Blocks opt into a lifecycle interface:

```ts
interface FinalizableBlock {
  isTranscriptBlockFinalized?(): boolean;        // false = still mutating
  getTranscriptBlockVersion?(): number;          // monotonic; catches post-finalize mutations
  getTranscriptBlockSettledRows?(): number;      // leading rows provably byte-stable NOW
  isDisplaceableBlock?(): boolean;               // retractable snapshot (todo/poll card)
  seal?(): void;                                 // freeze a displaceable in place
}
```

Behavior:
- Seam = leading run of finalized blocks **plus** the first live block's `getTranscriptBlockSettledRows()`. This is what lets a long streaming reply's already-scrolled-off head reach terminal history *mid-stream*.
- Assembly is **incremental**: `#lines: string[]` is persistent and mutated in place; per-block `BlockSegment { component, rawRef, contribution, width, generation, startRow, rowCount, sep, finalized, version }` records placement. A block whose `render()` returned the same array ref at an unchanged offset reuses its assembled rows; truncate + re-push starts at the first divergent block. The stable prefix is reported through `getRenderStablePrefixRows()` (reading consumes and re-bases it).
- `stripPlainBlankEdges(lines)` trims plain-blank leading/trailing rows from every block — **the container owns the gaps** (exactly one blank separator). Background-colored padding rows contain ANSI bytes so they survive: a plain blank is a row with no non-whitespace character at all.
- `isBlockUncommitted(component)` — must be checked before retracting an ephemeral block; rows already on the tape cannot be interior-deleted, so the block must be `seal()`ed instead.
- `isBlockInLiveRegion(component)` — self-animating finalized blocks poll this to stop animating once they sit above the seam.
- `renderViewportTail(width, maxRows)` — walks blocks bottom-up, stops once enough rows are collected, and is state-isolated (must not touch the persistent lines/segments/diff snapshots).
- `setNativeScrollbackCommittedRows(rows)` maps the committed contribution back into each child's raw render coordinates (accounting for trimmed blank edges) so nested containers can split correctly.

### 3.2 Assistant messages

`packages/coding-agent/src/modes/components/assistant-message.ts` — `class AssistantMessageComponent extends Container` holding a content container plus a marker slot.

`getTranscriptBlockSettledRows()` returns 0 when: finalized, not in transient/streaming mode, the message contains mermaid source (mermaid resolves asynchronously and can re-layout rows that looked settled), or a cache-invalidation marker sits above the content. Otherwise it walks children summing `child.render(width).length` for byte-stable children (`Markdown` via its frozen prefix, `Spacer`) and **stops at the first non-stable child** (animated thinking pulse, images, extension components, error rows).

`updateContent(message, transient)` is the streaming entry point; `invalidate()` drops fast-path children so theme changes rebuild with the current theme.

### 3.3 Markdown streaming — the frozen-prefix trick (the key perf idea)

`packages/tui/src/components/markdown.ts` (~3000 lines), `class Markdown implements Component, NativeScrollbackCommittedRows, NativeScrollbackReplay`. Uses `marked` with custom extensions (inline/block math, math environments, bounded autolink, custom hr, strict strikethrough).

- `#streamPrefixText` / `#streamPrefixTokens` / `#streamPrefixLineCache`. marked has no resumable lexer, but block tokenization is local across a blank-line boundary with balanced fences, so lex(prefix) ++ lex(tail) === lex(prefix + tail).
- `#lexTokens(text)`: if the text only *grew* and the old prefix still prefixes it, re-lex only the tail. Turns quadratic streaming reveal into linear. Falls back to a full lex on non-append edits, reference-link definitions (`HAS_REF_DEF` — they resolve document-wide), or CR presence (marked normalizes CRLF, desyncing raw-span offsets). Every fallback is correctness-preserving.
- `#freezeStablePrefix` uses `stableBlockBoundary(text, base, tokens)` to find the largest run of leading blocks ending on a hard blank line. The break must sit *inside* the text (end-of-text is deferred, more may arrive) and the next char must start real block content. `listMayContinueAt` conservatively refuses to freeze when marked could continue a list past the cut.
- `lexWindowed` / `lexDocument` window large documents (`WINDOWED_LEX_MIN_BYTES`): marked's list tokenizer runs the hr and lheading rules per list line against the remaining source, which is quadratic (66% of a streaming profile). Sticky-regex clones with lastIndex pinned to 0, plus linear charCode gates `urlTokenPossible` / `lheadingPossible` that never reject a src the built-in rule would match.
- `getLastRenderSettledRows()` exposes paddingY + frozen prefix rows, **hard-monotone within a text lineage**: a rewind resets to 0 and re-earns on the new lineage.
- `setNativeScrollbackCommittedRows(rows)` locks table column widths already on the tape so a later streamed row cannot retroactively widen them.
- Frozen code blocks are syntax-highlighted even in transient mode so their bytes byte-match the finalized render (amortized by the prefix line cache); the volatile tail stays unhighlighted, except streaming diff fences which line-highlight completed rows.
- Two-level render cache: L1 `#cachedLines` keyed on `RenderSignature { width, paddingX, paddingY, ... }`, L2 `renderCache` LRU; `clearRenderCache()` on theme change.
- `setText` has an **equality guard** — streaming re-emits identical text on no-delta ticks (throttled provider frames, reconciled tool updates), and without it the full lex + wrap runs per re-emit. This was a top CPU hotspot (issue #4353).
- `MarkdownTheme` interface: `heading, link, linkUrl, code, codeBlock, codeBlockBorder, quote, quoteBorder, hr, listBullet, bold, italic, underline, strikethrough` plus `DefaultTextStyle { color?, background? }`.

### 3.4 Tool calls

`packages/coding-agent/src/modes/components/tool-execution.ts` (`ToolExecutionComponent`) mounts per-tool renderers:

```ts
renderCall(args, options, theme): Component
renderResult(result, options, theme, args?): Component
// options: { expanded: boolean; isPartial: boolean; spinnerFrame?: number }
```

Contracts in `src/extensibility/custom-tools/types.ts` and `src/extensibility/extensions/types.ts`. Tool cards use viewport *pinning* for replacing preview/dashboard states. Frames come from `renderOutputBlock` / `renderStatusLine` (section 1.10).

Related block components: `user-message.ts`, `custom-message.ts`, `skill-message.ts`, `hook-message.ts`, `compaction-summary-message.ts`, `diff.ts`, `read-tool-group.ts`, `bash-execution.ts`, `eval-execution.ts`, `chat-block.ts`, `message-frame.ts`, `error-banner.ts`, `todo-reminder.ts`.

### 3.5 Minimal line usage — the actual techniques

1. Status line lives **inside the editor top border** — 0 extra rows.
2. Container owns inter-block gaps; blocks are edge-trimmed so no double blanks.
3. HUDs (`AnchoredLiveContainer`) rebuild in place and never commit, so no duplicate history rows.
4. Editor auto-sizes to content within `computeEditorMaxHeight`.
5. Image text fallback is height-preserving so demotion never shifts layout.
6. Tool results collapse by default; `app.tools.expand` (Ctrl+O) toggles; `app.tools.toggleVisibility` (Ctrl+Shift+O) hides tool activity entirely.

---

## 4. Input handling

### 4.1 Key parsing

`packages/tui/src/keys.ts`:

```ts
export type KeyId = BaseKey | ModifiedKeyId<BaseKey>;   // "ctrl+shift+p", "alt+left", ...
export const Key = { escape, enter, tab, up, down, ... } as const;
export function matchesKey(data: string, keyId: KeyId): boolean
export function parseKey(data: string): string | undefined
export function isKeyRelease(data) / isKeyRepeat(data)          // Kitty protocol only
export function parseKittySequence(data): ParsedKittySequence | null
export function setKittyProtocolActive(b) / isKittyProtocolActive()
export function extractPrintableText(data) / decodePrintableKey(data)
export function matchesRawBackspace(data, expectedModifier)     // 0x7f vs 0x08 disambiguation
export function isWindowsTerminalSession(): boolean
```

Parsing delegates to native (`matchesKeyNative`, `parseKeyNative`). The Kitty keyboard protocol gives release/repeat events and unambiguous modifiers; `modifyOtherKeys` is the fallback. Related: `bracketed-paste.ts`, `mouse.ts` (`parseSgrMouse`), `kill-ring.ts` (emacs yank / yank-pop), `stdin-buffer.ts`.

### 4.2 Keybindings registry

`packages/tui/src/keybindings.ts`:

```ts
export interface KeybindingDefinition { defaultKeys: KeyId | KeyId[]; description?: string }
export type KeybindingDefinitions = Record<string, KeybindingDefinition>;
export type KeybindingsConfig    = Record<string, KeyId | KeyId[] | undefined>;
export const TUI_KEYBINDINGS = { /* tui.editor.*, tui.input.*, tui.select.* */ };

export class KeybindingsManager {
  constructor(definitions: KeybindingDefinitions, userBindings: KeybindingsConfig = {});
  matches(data: string, keybinding: Keybinding): boolean;
  matchesCanonical(canonical: string | undefined, kb: Keybinding): boolean;  // hot path
  getDisplayString(id): string;
  conflicts: KeybindingConflict[];       // { key, keybindings[] } for user double-claims
}
export function canonicalKeyId(key: string): string   // modifier order: ctrl, shift, alt, super
export function addKeyAliases(keys: Set<string>, key: KeyId): void
```

`canonicalKeyId` normalizes: case-insensitive, `esc` -> `escape`, `return` -> `enter`, an ASCII uppercase base implies `shift`, modifiers sorted. `SHIFTED_SYMBOL_KEYS` get a `shift+` alias.

The `Keybindings` interface is **extended by declaration merging** from downstream packages, so `packages/coding-agent` adds its `app.*` ids to the same typed registry. Good pattern.

Built-in TUI actions: `tui.editor.*` (cursorUp, cursorDown, cursorLeft, cursorRight, cursorWordLeft, cursorWordRight, cursorLineStart, cursorLineEnd, jumpForward, jumpBackward, pageUp, pageDown, deleteCharBackward, deleteCharForward, deleteWordBackward, deleteWordForward, deleteToLineStart, deleteToLineEnd, yank, yankPop, undo), `tui.input.*` (newLine, submit, tab, copy), `tui.select.*` (up, down, pageUp, pageDown, confirm, cancel). Defaults are readline/emacs-flavored: ctrl+a, ctrl+e, ctrl+w, ctrl+k, ctrl+u, ctrl+y, alt+y; shift+enter or ctrl+j for newline; enter to submit.

**User config**: `~/.omp/agent/keybindings.yml` — a flat YAML map of action id to chord string or array of chords. An empty array disables the action. Profiles inherit the default profile's file, overridden action by action. App defaults are in `docs/keybindings.md` (Ctrl+P cycle model, Alt+M model select, Shift+Tab thinking level, Ctrl+R history search, Ctrl+O expand tools, Ctrl+T thinking visibility, Ctrl+G external editor, Ctrl+Q or Ctrl+Enter queue follow-up, Alt+L display reset, Alt+A agent hub, Ctrl+L live voice). `/hotkeys` prints the active set.

### 4.3 Multiline editor

`packages/tui/src/components/editor.ts` — `class Editor implements Component, Focusable`. Subclassed as `CustomEditor` in `packages/coding-agent/src/modes/components/custom-editor.ts`.

State: `#state: EditorState { lines: string[]; cursorLine: number; cursorCol: number; ... }` — a plain line array, not a rope or gap buffer.

```ts
export interface EditorTheme { borderColor: (s: string) => string; selectList: SelectListTheme; symbols: SymbolTheme; ... }
export interface EditorTopBorder { content: string; width: number }
export interface HistoryStorage { add(prompt, cwd?): Promise<void>; getRecent(limit): HistoryEntry[] }

// text
getText() / getExpandedText() / getLines() / setText(t) / insertText(t) / insertPaste(content)
deleteBeforeCursor(count) / setVolatileText(t) / clearVolatileText()
getCursor() / moveToLineStart() / moveToLineEnd() / moveToMessageStart() / moveToMessageEnd()
// layout
setMaxHeight(n) / setPaddingX(n) / setBorderVisible(b) / setPromptGutter(s)
setScrollbarVisible(b) / setTopBorder(c) / setTopBorderProvider(fn)
getTopBorderAvailableWidth(terminalWidth)
// cursor / IME
setUseTerminalCursor(b) / getUseTerminalCursor() / setImeSafeCursorLayout(b)
cursorOverride, cursorOverrideWidth                       // e.g. the mic glyph during STT
// autocomplete
setAutocompleteProvider(p: AutocompleteProvider)
setAutocompleteMaxVisible(n)          // clamped to [3, 20]
onAutocompleteUpdate, onAutocompleteCancel
// history
setHistoryStorage(s: HistoryStorage)  // seeds history from getRecent(100)
// misc
borderColor, onEscape, setShimmerRepaintHandler(fn)
pendingImages / pendingImageLinks / imageLinks
```

Present: soft-wrapped multiline with scroll offset, undo stack (`#recordUndoState`, `#withUndoSuspended`), emacs kill-ring, bracketed paste handled **iteratively** so a fragmented paste stream cannot grow the call stack, paste markers (long pastes collapse to a chip, `#expandPasteMarkers` restores on submit), pending images, external-editor handoff (`openInEditor` + `getEditorCommand` from `utils/external-editor.ts`), right-border scrollbar, IME-safe bottom border.

### 4.4 Autocomplete and slash commands

`packages/tui/src/autocomplete.ts` (1078 lines):

```ts
export interface AutocompleteItem { value: string; label: string; description?: string; hint?: string }

export interface SlashCommand {
  name: string; aliases?: string[]; description?: string; argumentHint?: string;
  allowArgs?: boolean;
  getAutocompleteDescription?(): string | undefined;         // sync, side-effect free
  getArgumentCompletions?(argumentPrefix): Awaitable<AutocompleteItem[] | null>;
  getInlineHint?(argumentText): string | null;
}

export interface AutocompleteProvider {
  getSuggestions(lines, cursorLine, cursorCol): Promise<{ items; prefix } | null>;
  applyCompletion(lines, cursorLine, cursorCol, item, prefix):
      { lines; cursorLine; cursorCol; onApplied?() };
  getInlineHint?(lines, cursorLine, cursorCol): string | null;
  trySyncSlashCompletion?(textBeforeCursor): { items; prefix } | null;
  trySyncInlineReplace?(textBeforeCursor): { replaceLen: number; insert: string } | null;
  getForceFileSuggestions?(lines, l, c): Promise<{ items; prefix } | null>;   // Tab
  shouldTriggerFileCompletion?(lines, l, c): boolean;
}

export function findLeadingSlashCommandStart(text): number | null
export function findTrailingSlashCommandStart(text): number | null
export function scoreCommandTextMatch(lowerPrefix, lowerTarget): number
  // exact 1000 > prefix 900 (flat, so registry order breaks ties) > fuzzy subsequence 1..40
```

- The **sync/async split matters**: `trySyncSlashCompletion` and `trySyncInlineReplace` run on every keystroke with zero I/O so Enter applies deterministically; the async `getSuggestions` does file discovery.
- File completion uses the native fuzzy finder `fuzzyFind` from `@oh-my-pi/pi-natives` with profile `{ maxResults: 100, hidden: true, gitignore: true, cache: true }`.
- Path prefixes handle at-mentions and quoting; the delimiter set is space, tab, double quote, single quote, and equals.
- Inline **ghost hint** text renders dim after the cursor.
- Coding-agent stacks extra providers: `modes/prompt-action-autocomplete.ts`, `internal-url-autocomplete.ts`, `emoji-autocomplete.ts`, `github-ref-autocomplete.ts`. Extensions wrap the base provider via `addAutocompleteProvider(factory)` (`interactive-mode.ts` ~1294), applied in registration order; a factory that throws or returns a malformed provider is skipped so one broken extension cannot kill core autocomplete.

---

## 5. TUI to agent-core wiring

**Event subscription. Not JSON-RPC, not a virtual DOM, not callbacks-per-widget.**

`packages/coding-agent/src/modes/controllers/event-controller.ts`:

```ts
type AgentSessionEventKind = AgentSessionEvent["type"];
type AgentSessionEventHandlers = {
  [E in AgentSessionEventKind]: (event: Extract<AgentSessionEvent, { type: E }>) => Promise<void>;
};

export class EventController {
  #handlers: AgentSessionEventHandlers;              // exhaustive, checked with satisfies
  subscribeToAgent(): void {
    this.ctx.unsubscribe = this.ctx.session.subscribe(async (event: AgentSessionEvent) => {
      await this.handleEvent(event);
    });
  }
  async handleEvent(event: AgentSessionEvent): Promise<void> {
    if (!this.ctx.isInitialized) await this.ctx.init();
    const run = this.#handlers[event.type] as (e: AgentSessionEvent) => Promise<void>;
    await run(event);
  }
}
```

Event kinds in the handler table: `agent_start`, `agent_end`, `message_start`, `message_update`, `message_end`, `turn_end`, `tool_execution_start`, `tool_execution_update`, `tool_execution_end`, `auto_compaction_start/end`, `auto_retry_start/end`, `retry_fallback_applied/succeeded`, `ttsr_triggered`, `todo_reminder`, `todo_auto_clear`, `irc_message`, `notice`, `goal_updated`.

Per-event pattern: mutate or create a component, then `this.ui.requestRender()`. `message_update` finds `ctx.streamingComponent` (an `AssistantMessageComponent`) and calls `updateContent(message, transient = true)`. Tool events index `ctx.pendingTools: Map<toolCallId, ToolExecutionHandle>`.

Ordering hazards the code explicitly handles (read the comments before porting):
- A superseded `agent_end` can arrive *after* the next turn's `agent_start`, because dispatch crosses an async extension-emit hop in `AgentSession#emitSessionEvent`. Running teardown then would kill the live turn's loader.
- Retry sagas coalesce many `agent_end` events into one, so a single `agent_end` may be the final failure rather than an intermediate blip; only the retry lifecycle events distinguish them.
- Orphaned `tool_execution_end` events are buffered in `#orphanedToolCompletions`; retracted tool call ids are tracked so aborted-tool synthetic start/end pairs do not recreate cards.

Other controllers, `src/modes/controllers/`: `input-controller.ts` (key handlers + editor submit), `command-controller.ts`, `mcp-command-controller.ts`, `selector-controller.ts`, `extension-ui-controller.ts`, `streaming-reveal.ts`, `session-focus-controller.ts`, `btw-controller.ts`, `todo-command-controller.ts`.

Extension / tool UI mounting (`docs/tui.md`):

```ts
ctx.ui.custom<T>(
  (tui: TUI, theme: Theme, keybindings: KeybindingsManager, done: (r: T) => void) => Component,
  options?: { overlay?: boolean },
): Promise<T>
```

`done()` is mandatory. Guard with `ctx.hasUI` — RPC mode returns undefined-as-never, headless is a no-op. Known bug documented there: `HookUIContext.custom` is *typed* 3-arg but *called* 4-arg by `extension-ui-controller.ts`.

Alternative front ends reuse the same session events: `src/modes/rpc/` (JSON-RPC), `src/modes/acp/` (Agent Client Protocol), `src/modes/print-mode.ts`. **This is the property to preserve.**

---

## 6. MCP

Hand-rolled client. Protocol version `2025-03-26`, client info `{ name: "omp-coding-agent", version: "1.0.0" }` (`src/mcp/client.ts`).

### 6.1 Config

Files (`docs/mcp-config.md`):
- Project: `.omp/mcp.json` (also reads `.omp/.mcp.json`)
- User: `~/.omp/agent/mcp.json`, or `~/.omp/profiles/NAME/agent/mcp.json` under a named profile
- Root fallbacks: `mcp.json`, `.mcp.json`
- **Imported from other tools**: `~/.claude.json`, `~/.claude/mcp.json`, `.claude/.mcp.json`, `~/.codex/config.toml` (`[mcp_servers.*]`), `~/.gemini/settings.json`, `opencode.json`, `~/.cursor/mcp.json`, `~/.codeium/windsurf/mcp_config.json`, `.vscode/mcp.json` (`mcp.servers`), Claude marketplace plugins, OMP extension packages.

File shape:
```json
{
  "mcpServers": { "name": { "type": "stdio", "command": "npx", "args": ["-y", "pkg"] } },
  "disabledServers": ["name"],
  "enabledServers":  ["name"]
}
```
Plus an optional JSON-Schema URL key. `disabledServers` (denylist) always wins over `enabledServers` (allowlist), which itself overrides a source's `enabled: false`.

Types — `src/mcp/types.ts`:
```ts
interface MCPServerConfigBase {
  enabled?: boolean;
  timeout?: number;                          // ms; 0 disables client-side timeout
  requestIdFormat?: "string" | "number";     // OMP-only; "string" = snowflake ids
  auth?: MCPAuthConfig;
  oauth?: { clientId?; clientSecret?; redirectUri?; callbackPort?; callbackPath?; prompt? };
}
interface MCPStdioServerConfig extends Base { type?: "stdio"; command: string; args?; env?; cwd? }
interface MCPHttpServerConfig  extends Base { type: "http"; url: string; headers? }
interface MCPSseServerConfig   extends Base { type: "sse";  url: string; headers? }
type MCPServerConfig = MCPStdioServerConfig | MCPHttpServerConfig | MCPSseServerConfig;

interface MCPAuthConfig { type: "oauth" | "apikey"; credentialId?; tokenUrl?; clientId?; clientSecret?; resource? }
interface MCPAuthChallenge { readonly wwwAuthenticate: readonly string[] }   // from mcp/www_authenticate meta
```
Note: `apikey` is accepted but **not implemented** — put API keys in stdio `env` or remote `headers` using env-var or bang-command indirection.

Timeout precedence: `OMP_MCP_TIMEOUT_MS` (process-wide) -> `config.timeout` -> 30 s default; `0` disables. `src/mcp/timeout.ts`: `resolveMCPTimeoutMs`, `isMCPTimeoutEnabled`, `describeMCPTimeout`, `createMCPTimeout`, `getNeverAbortSignal`.

Loading — `src/mcp/config.ts`:
```ts
export async function loadAllMCPConfigs(cwd, options?: {
  enableProjectConfig?; filterExa?; filterBrowser?
}): Promise<{ configs: Record<string, MCPServerConfig>; exaApiKeys: string[]; sources: Record<string, SourceMeta> }>
export function validateServerConfig(name, config): string[]
export function isExaMCPServer(name, config) / extractExaApiKey(config) / filterExaMCPServers(...)
```
It runs through the generic capability/discovery system — `loadCapability(mcpCapability.id, { cwd, filter, suppress })` — then `convertToLegacyConfig`. Note the **filter vs suppress** distinction: *filtered* entries are dropped before dedupe (a project entry with project config disabled must not shadow anything); *suppressed* entries still own their name at key-level dedupe (a disabled project `foo` keeps a lower-priority user `foo` disabled) but never equivalence-shadow a differently-named enabled server.

Writer — `src/mcp/config-writer.ts`: `addMCPServer` / `updateMCPServer` / `readDisabledServers` / `readEnabledServers`. Server names allow letters, digits, underscore, dash, dot, colon; max 100 chars. (The bundled JSON schema omits colon — a known mismatch for namespaced plugin entries.)

### 6.2 Transports

`src/mcp/transports/`. Common interface (`types.ts`):
```ts
export interface MCPTransport {
  request<T>(method, params?, options?: { signal? }): Promise<T>;
  notify(method, params?): Promise<void>;
  close(): Promise<void>;
  readonly connected: boolean;
  onClose?: () => void;
  onError?: (e: Error) => void;
  onNotification?: (method: string, params: unknown) => void;
  onRequest?: (method: string, params: unknown) => Promise<unknown>;   // server -> client
}
```

**stdio** (`transports/stdio.ts`, 906 lines — most of it Windows hardening): newline-delimited JSON over `Bun.spawn` pipes. `StdioSpawnCommand { cmd, windowsHide?, detached, windowsVerbatimArguments? }`. Platform rules, each with a documented reason:
- Linux / other POSIX: `detached: true` (setsid) so terminal job control (Ctrl+Z SIGTSTP, background-read SIGTTIN) cannot stop stdio servers and leave the read loop blocked on silent pipes.
- macOS: `detached: false` — LaunchServices/TCC attributes Apple Events automation to the responsible terminal only while the child stays in the inherited session (issue #4987).
- Windows: `detached: false`; `windowsHide` only when the host has no inheritable console (else hidden wrapper chains spawn visible conhost windows, #3567); `.cmd` / `.bat` launches go through `cmd.exe /d /e:ON /v:OFF /c` with **BatBadBut (CVE-2024-24576) escaping** via `escapeCmdQuotedInterior` (percent neutralized, quotes doubled, backslash runs doubled before quotes) plus `windowsVerbatimArguments` so libuv does not re-quote. NUL/CR/LF in a token is rejected outright. npm cmd-shims whose fallback interpreter is node are bypassed straight to node plus target plus args.

**http** (`transports/http.ts`, Streamable HTTP, 526 lines): `class HttpTransport implements MCPTransport`. POST for requests; `Mcp-Session-Id` tracked from the initialize response; `startSSEListener()` opens a GET SSE stream for server-initiated messages, bounded by `resolveSSEConnectTimeoutMs(configTimeout)` (capped at `HTTP_SSE_CONNECT_TIMEOUT_MS = 1000` and at a quarter of the request timeout). `onAuthError` is the token-refresh hook fired on 401/403, returning updated headers or null.

**sse** (`transports/sse.ts`): legacy dual-endpoint SSE. Deprecated by the spec in favor of `http`, kept for compatibility.

Request ids: `src/mcp/request-id.ts` `RequestIdAllocator` — per-transport monotonic integers by default, snowflake strings when `requestIdFormat` is `string`.

### 6.3 Connection

`src/mcp/client.ts`:
```ts
export async function connectToServer(name, config, options?: {
  signal?; onNotification?; onRequest?
}): Promise<MCPServerConnection>
export async function listTools(connection, options?): Promise<MCPToolDefinition[]>   // paginated by nextCursor
export async function callTool(connection, toolName, args, options?): Promise<MCPToolCallResult>
export async function disconnectServer(connection): Promise<void>
export async function listResources / listResourceTemplates / readResource / subscribeToResources
export async function listPrompts / getPrompt
export function serverSupportsTools(caps) / serverSupportsResources / serverSupportsPrompts
```

Handshake order matters: `initialize` (advertising `capabilities: { roots: { listChanged: false } }`) -> **for HTTP, open the SSE listener** via the `onInitialized` hook, *before* `notifications/initialized`, so a server-triggered `roots/list` can be delivered -> `notifications/initialized`. `defaultRequestHandler` answers `ping` with an empty object and `roots/list` with the project dir as a file URL plus its basename (read at call time so a cwd change is reflected); anything else throws with `code: -32601`. The transport is closed on init failure, and again if `withTimeout` rejects while `connect()` is still pending.

```ts
export interface MCPServerConnection {
  name; config; transport; serverInfo; capabilities;
  tools?; resources?; resourceTemplates?; prompts?; instructions?; _source?: SourceMeta;
}
```

### 6.4 Manager lifecycle

`src/mcp/manager.ts` — `export class MCPManager`, `constructor(cwd: string, toolCache: MCPToolCache | null = null)`, plus a `static instance` for internal URL handlers.

Registries:
```
#connections          Map<string, MCPServerConnection>
#pendingConnections   Map<string, Promise<MCPServerConnection>>
#pendingToolLoads     Map<string, Promise<{ connection, serverTools }>>
#pendingReconnections Map<string, Promise<MCPServerConnection | null>>
#serverConfigs        Map<string, MCPServerConfig>    // UNRESOLVED, so reconnect re-resolves creds
#sources              Map<string, SourceMeta>
#tools                CustomTool[]                     // stable name order
#reconnectHistory     Map<string, number[]>  +  #epoch
```

Public API:
```ts
discoverAndConnect(options?: MCPDiscoverOptions): Promise<MCPLoadResult>
getTools() / getConnection(name) / getConnectionStatus(name): "connected" | "connecting" | "disconnected"
getSource(name) / getServerConfig(name) / getConnectedServers() / getAllServerNames()
waitForConnection(name): Promise<MCPServerConnection>
prepareConfig(config, { oauth? }): Promise<MCPServerConfig>
disconnectServer(name) / disconnectAll()
refreshServerTools(name) / refreshAllTools() / refreshServerResources / refreshServerPrompts
ensureServerResources(name) / getServerResources(name) / getServerPrompts(name)
getServerInstructions(): Map<string, string>
setAuthStorage(s) / setAuthHandler(h: MCPAuthHandler) / setNotificationsEnabled(b)
setOnToolsChanged(h) / setOnResourcesChanged(h) / setOnPromptsChanged(h)
addNotificationListener(l): () => void
```

**Fast startup gate — the single best idea here.** `connectServers()` races `Promise.allSettled(all connect + tools/list tasks)` against `delay(STARTUP_TIMEOUT_MS = 250)`. After 250 ms:
- settled tasks become live `MCPTool` instances;
- rejected tasks become per-server error entries (others continue);
- still pending **with a tool-cache hit** become `DeferredMCPTool.fromTools(name, cached, () => waitForConnection(name), source, reconnect)` — the tool exists and is callable before the server is up;
- still pending **without cache** contribute nothing at startup; a background continuation replaces that server's tool slice, restores name ordering, and fires `#onToolsChanged` when it lands (a slow server no longer blocks startup, issue #2100).

Cache: `src/mcp/tool-cache.ts` `MCPToolCache`, backed by `AgentStorage` (SQLite `agent.db`). Best-effort; unavailability degrades to no cache.

**Reconnect** is wired to `transport.onClose` (never polled — there is no health monitor). Backoff 500, 1000, 2000, 4000 ms. Circuit breaker `#tripReconnectBreaker`: sliding window `RECONNECT_BURST_WINDOW_MS = 30_000`, `RECONNECT_BURST_LIMIT = 5` -> suspends automatic reconnects and tears down the stale connection while leaving tools registered; manual `/mcp reconnect` resets the history. `#epoch` increments on `disconnectAll()` so an in-flight reconnect that finishes later cannot resurrect a dead connection.

**Server-initiated notifications**: `notifications/tools/list_changed` -> `refreshServerTools`; `resources/list_changed` -> `refreshServerResources`; `resources/updated` -> `#onResourcesChanged` (subscribed URIs only); `prompts/list_changed` -> `refreshServerPrompts`. Every frame (known and server-custom) is then fanned out to listeners with per-listener error isolation. With no listener attached, up to 100 frames buffer (drop-oldest) and drain into the first subscriber.

**Auth resolution**: `#resolveAuthConfig(config, { forceRefresh?, oauth? })` resolves managed OAuth credentials into an `Authorization` header and performs env-var / bang-command shell substitution across `env` and `headers`. Only the *unresolved* config is stored in `#serverConfigs`, so resolved tokens never leak into reconnect state.

**Ownership**: a top-level session owns the manager it creates; `AgentSession.dispose()` disconnects it with a 3-second cleanup timeout. A subagent given `options.mcpManager` **borrows** the parent's and must not disconnect it (`src/task/executor.ts`). `/mcp reload` deliberately reuses the manager object after `disconnectAll()` so installed callbacks survive.

### 6.5 OAuth

`src/mcp/oauth-flow.ts` (831 lines) + `oauth-discovery.ts` + `oauth-credentials.ts`.

```ts
export class MCPOAuthFlow {
  constructor(config: MCPOAuthConfig, options: { onAuthUrl(url); onManualInput(); signal? });
  login(): Promise<OAuthCredentials>;
  resolvedClientId?: string;
}
export interface MCPOAuthConfig {
  authorizationUrl; tokenUrl; registrationUrl?; clientId?; clientSecret?; scopes?;
  prompt?; redirectUri?; callbackPort?; callbackPath?; resource?; serverUrl?; stripSameOriginResource?;
}
export function mcpOAuthCredentialId(serverUrl, profile = getActiveProfile()): string
  // -> "mcp_oauth:profile:<profile>:<serverUrl>"
export function isManagedMCPOAuthCredentialId(id) / mcpOAuthCredentialProfile(id)
export interface MCPStoredOAuthCredential extends OAuthCredential {
  tokenUrl?; clientId?; clientSecret?; resource?; authorizationUrl?;
}
```

Mechanics:
- Loopback HTTP callback server (`@oh-my-pi/pi-ai/oauth/callback-server` -> `OAuthCallbackFlow`), default port 3000, path `/callback`. Port fallback is allowed **only** when no client_id is statically known (`staticClientIdFromConfig` checks config and the authorize URL query): a pinned client_id was registered against a specific redirect URI, so silently advertising a different port is rejected by providers like Atlassian (500 in the browser, local flow hangs to timeout). HTTPS loopback redirects require a distinct `callbackPort` behind the TLS terminator.
- **PKCE + Dynamic Client Registration (RFC 7591)** when no client_id is known: `#tryRegisterClient` against `registrationUrl`. DCR-issued secrets stay embedded in the stored credential and are never written into (possibly committed) config files.
- **Discovery**: `discoverOAuthEndpoints(serverUrl, authServerUrl?, resourceMetadataUrl?, { protectedScopes? })`, `analyzeAuthError(error, url?)` (parses `WWW-Authenticate` and RFC 9728 protected-resource metadata), `fetchResourceMetadataScopes(url)` for the JSON-error-body path.
- `prompt` is omitted by default, except that requesting `offline_access` defaults it to `consent` (OIDC Core requires it for refresh access). Resource indicators: `filterResourceIndicator(resource, serverUrl, { stripSameOriginResource })` drops OMP-synthesized same-origin fallbacks but preserves provider-advertised values (which may legitimately be origin-only or path-scoped).
- **Credential binding is per-profile, URL-keyed.** A project `mcp.json` with *no* `auth` block still resolves each profile's own credential automatically, including under a shared auth broker. `/mcp reauth` on a definition-only entry leaves the file untouched. An explicitly configured `Authorization` header always wins. Security note carried in the docs: committed MCP definitions are trusted input (stdio entries already run arbitrary commands), so review a repo's `mcp.json` before opening it under a credentialed profile.
- **Tool-level auth challenge**: a tool result carrying the `mcp/www_authenticate` meta key produces an `MCPAuthChallenge`, which triggers `MCPAuthHandler` -> reauth -> reconnect -> one retry. Servers that allow an anonymous handshake but protect individual tool calls are handled by this path only.

UI flow: `src/modes/controllers/mcp-command-controller.ts` — `#handleOAuthFlow(...)` with a 5-minute timeout raced against an `AbortController`; Esc yields `MCPOAuthCancelledError` (distinct from a timeout so the UI shows cancelled rather than an error banner); headless fallback via `oauthManualInput` and `/login VALUE`; the URL component renders the full authorization URL hard-wrapped into width-fitted rows so trailing OAuth params can never be silently lost on copy (issue #4418). `#persistOAuthResult` folds the result back into config: the `auth` block records the credential pointer plus refresh material, `oauth` echoes only the client id, and only a *user-supplied* client secret is ever written.

Slash commands: `/mcp add|list|test|reauth|unauth|enable|disable|reconnect|reload|smithery-search|smithery-login|smithery-logout`. Smithery registry integration in `smithery-registry.ts`, `smithery-connect.ts`, `smithery-auth.ts`. Wizard UI: `src/modes/components/mcp-add-wizard.ts`.

### 6.6 Tool registration into the agent

`src/mcp/tool-bridge.ts` (692 lines):

```ts
export function createMCPToolName(serverName, toolName): string
// sanitize each part: lowercase, non [a-z_] runs -> "_", collapse runs, trim edge underscores
// strip a redundant server prefix: server "puppeteer" + tool "puppeteer_screenshot"
//   -> "mcp__puppeteer_screenshot", NOT "mcp__puppeteer_puppeteer_screenshot"
// final shape: "mcp__" + server + "_" + tool

export class MCPTool         implements CustomTool<TSchema, MCPToolDetails> { ... }
export class DeferredMCPTool implements CustomTool<TSchema, MCPToolDetails> { ... }  // awaits waitForConnection

export interface MCPToolDetails {
  serverName; mcpToolName; isError?; rawContent?: MCPContent[];
  mcpMeta?: Record<string, unknown>; provider?; providerName?; meta?: OutputMeta;
}
export type MCPReconnect = (o?: { authChallenge?: MCPAuthChallenge }) => Promise<MCPServerConnection | null>;
export function isRetriableConnectionError(error): boolean
```

Each tool sets its name from `createMCPToolName(...)`, a label of server-slash-tool, a description, and `parameters = normalizeSchemaForMCP(tool.inputSchema)`.

Outbound arg pipeline (`prepareOutboundArgs`) — three cleanups worth copying verbatim:
1. `stripHarnessIntent` — removes the harness-injected `i` field (`INTENT_FIELD`) unless the server's own `inputSchema.properties` declares it. Strict-schema servers (Linear, anything with `additionalProperties: false` or Zod strict) reject every call carrying it. The MCP boundary is the authoritative guard so no caller has to pre-strip.
2. `omitUnusedOptionalArgs` — drops non-required keys whose value is undefined, empty string, or an empty object.
3. `resolveOutboundLocalUrlArgs` — rewrites `local://` URLs into real filesystem paths external servers can read; cycle-safe via a `WeakSet`, and allocation-free when nothing changes.

Result handling: `formatMCPContent(content)` flattens text, image placeholders, and resource URIs plus text. Errors become `MCP error: MESSAGE` tool content (abort stays abort). `isRetriableConnectionError` matches econnrefused, econnreset, epipe, enetunreach, ehostunreach, fetch failed, transport not connected, transport closed, network error, plus HTTP 404/502/503 (stale session after a server restart) -> one reconnect + one retry.

Wiring into a session:
- `src/mcp/loader.ts` `discoverAndLoadMCPTools(cwd, opts)` returns `{ manager, tools: LoadedCustomTool[], errors, connectedServers, exaApiKeys }`, decorating each path as `mcp:SERVER via PROVIDER`. A hard discovery failure returns empty tools plus one synthetic `.mcp.json` error rather than failing session startup.
- `src/sdk.ts` `createAgentSession()`: headless/SDK awaits it; interactive (`hasUI: true`) constructs `MCPManager` up front and defers `discoverAndConnect()` to a background task started after the session exists, binding via `session.refreshMCPTools(...)` (disposing the manager if the session was torn down mid-connect). `discoverMCPServers()` is the public convenience wrapper.
- `AgentSession.refreshMCPTools(tools)` (`src/session/agent-session.ts`) removes all `mcp__` tools, re-wraps the latest set, and re-activates — live rebinding without restart. `setOnToolsChanged` routes late connections, `tools/list_changed`, reconnects, and disconnects into the same path.
- Name collisions between distinct origins are logged and resolved by a **stable origin key** (server name + original tool name), never array order, so reconnect ordering cannot flip ownership.

Rendering of MCP calls/results: `src/mcp/render.ts` (`renderMCPCall`, `renderMCPResult`).

---

## 7. Key types and functions — index

### TUI (`packages/tui/src/`)
| Symbol | File |
|---|---|
| `Component`, `Focusable`, `Container`, `TUI`, `CURSOR_MARKER` | `tui.ts` |
| `NativeScrollbackLiveRegion`, `NativeScrollbackCommittedRows`, `NativeScrollbackReplay`, `RenderStablePrefix`, `ViewportTailProvider`, `OverlayFocusOwner` | `tui.ts` |
| `RenderScheduler`, `RenderIntent`, `findCommittedPrefixResync`, `coalesceAdjacentSgr` | `tui.ts` |
| `OverlayOptions`, `OverlayHandle`, `OverlayAnchor`, `SizeValue` | `tui.ts` |
| `visibleWidth`, `truncateToWidth`, `wrapTextWithAnsi`, `sliceWithWidth`, `replaceTabs`, `padding` | `utils.ts` |
| `matchesKey`, `parseKey`, `KeyId`, `isKeyRelease`, `Key` | `keys.ts` |
| `KeybindingsManager`, `TUI_KEYBINDINGS`, `canonicalKeyId` | `keybindings.ts` |
| `AutocompleteProvider`, `AutocompleteItem`, `SlashCommand`, `scoreCommandTextMatch` | `autocomplete.ts` |
| `Editor`, `EditorTheme`, `EditorTopBorder`, `HistoryStorage` | `components/editor.ts` |
| `Markdown`, `MarkdownTheme`, `clearRenderCache`, `stableBlockBoundary` | `components/markdown.ts` |
| `SelectList`, `Text`, `Spacer`, `Loader`, `CancellableLoader`, `ScrollView`, `TabBar`, `Image` | `components/` |
| `TERMINAL`, `ImageProtocol`, `isInsideTerminalMultiplexer` | `terminal-capabilities.ts` |
| `ProcessTerminal`, `Terminal` | `terminal.ts` |
| `planDeccaraFills` | `deccara.ts` |
| `ImageBudget`, `encodeKittyDeleteImage` | `components/image.ts`, `kitty-graphics.ts` |
| `parseSgrMouse` | `mouse.ts` |

### Coding-agent integration
| Symbol | File |
|---|---|
| `InteractiveMode`, `computeEditorMaxHeight`, `AnchoredLiveContainer` | `src/modes/interactive-mode.ts` |
| `TranscriptContainer`, `FinalizableBlock`, `BlockSegment` | `src/modes/components/transcript-container.ts` |
| `AssistantMessageComponent` | `src/modes/components/assistant-message.ts` |
| `ToolExecutionComponent` | `src/modes/components/tool-execution.ts` |
| `StatusLineComponent`, `getTopBorder(width)` | `src/modes/components/status-line/component.ts` |
| `renderSegment`, `getSeparator`, `getPreset`, `StatusLineSegmentId` | `src/modes/components/status-line/` (segments, separators, presets, types) |
| `EventController`, `AgentSessionEventHandlers` | `src/modes/controllers/event-controller.ts` |
| `Theme`, `theme`, `setTheme`, `getMarkdownTheme`, `getEditorTheme`, `highlightCode` | `src/modes/theme/theme.ts` |
| `renderOutputBlock`, `CachedOutputBlock`, `framedBlock`, `renderStatusLine`, `Hasher` | `src/tui/` (output-block, status-line, utils) |

### MCP
| Symbol | File |
|---|---|
| `MCPServerConfig`, `MCPTransport`, `MCPServerConnection`, `MCPToolDefinition`, `MCPAuthChallenge` | `src/mcp/types.ts` |
| `loadAllMCPConfigs`, `validateServerConfig` | `src/mcp/config.ts` |
| `connectToServer`, `listTools`, `callTool` | `src/mcp/client.ts` |
| `MCPManager`, `STARTUP_TIMEOUT_MS`, reconnect-burst constants | `src/mcp/manager.ts` |
| `MCPTool`, `DeferredMCPTool`, `createMCPToolName`, `isRetriableConnectionError` | `src/mcp/tool-bridge.ts` |
| `MCPOAuthFlow`, `mcpOAuthCredentialId`, `MCPStoredOAuthCredential` | `src/mcp/oauth-flow.ts` |
| `discoverOAuthEndpoints`, `analyzeAuthError`, `fetchResourceMetadataScopes` | `src/mcp/oauth-discovery.ts` |
| `HttpTransport`, `createSseTransport`, `createStdioTransport` | `src/mcp/transports/` |
| `discoverAndLoadMCPTools` | `src/mcp/loader.ts` |
| `MCPToolCache` | `src/mcp/tool-cache.ts` |
| `RequestIdAllocator` | `src/mcp/request-id.ts` |
| `MCPCommandController` (the `/mcp` commands) | `src/modes/controllers/mcp-command-controller.ts` |
| JSON schema | `src/config/mcp-schema.json` |

---

## Porting notes to Python

### TUI stack recommendation

**Do not port the omp renderer.** Its value is a 4000-line append-only commit ledger solving one problem — native-scrollback fidelity across kitty, ghostty, Windows Terminal, tmux, and zellij — that took a documented war journal of failed policies to get right, and it is only trustworthy because a virtual-terminal stress harness with a shadow ledger validates it. Reimplementing that in Python without the harness reproduces the yank / flash / corruption / invisible-until-resize failure family.

**Recommended: Textual** (`textual` + `rich`), with one tradeoff acknowledged up front.

| Option | Verdict |
|---|---|
| **Textual** | Recommended. Widget tree + CSS-like styling maps onto the Component/theme split; async-native so it composes with an asyncio agent loop; ships `RichLog`, `TextArea` (multiline, undo, syntax), `Input` with suggesters, `OptionList`, modal screens, a real `BINDINGS` system, and its own theme / CSS-variable token system. Devtools and snapshot testing (`pytest-textual-snapshot`) are genuinely good. |
| **prompt_toolkit** | Only if a REPL-shaped prompt is the whole product. Its multiline `Buffer`, completers, and key-binding registry are excellent and it stays on the normal screen — but its full-screen layout engine is clunkier and less componentized. Best fallback if you want to keep omp's transcript-in-native-scrollback property. |
| **Raw ANSI** | No. Budget a person-quarter minimum and you still will not match omp's terminal matrix. |
| `rich` alone | No. `rich.live.Live` only repaints a bounded region and is not an input/focus framework. Fine for print/headless mode. |

**The one thing Textual costs you.** Textual runs in the **alt screen** by default, so you lose native scrollback, native mouse selection, and transcript-after-exit — exactly the properties omp organized its entire renderer around. Mitigations, best first:

1. **Accept it.** Give the transcript a `RichLog` or `VerticalScroll` with app-owned scrollback, plus explicit copy-transcript and export actions. Most Python TUIs do this. Also evaluate Textual's inline mode (`app.run(inline=True)`), which renders below the prompt without the alt screen but is more limited.
2. **Hybrid**: print/headless output goes through `rich` on the normal screen; the Textual app is only for interactive sessions.
3. Only if native scrollback is a hard product requirement: prompt_toolkit with a hand-managed append-only transcript.

### Framework-independent wins to port regardless

1. **The streaming markdown frozen prefix.** Naive re-render-the-whole-message-per-token is quadratic and is *the* dominant cost. Keep a (frozen_text, frozen_tokens, frozen_rendered_lines) triple, cut at the last blank-line block boundary with balanced fences, re-parse only the tail. `markdown-it-py` gives a token stream you can splice under the same conditions (no reference-link definitions, no CR, cut on a blank line). Cache the rendered `rich` segments for the frozen part. Add omp's `setText` equality guard — providers re-emit identical text on no-delta ticks.
2. **Finalized-block / settled-rows protocol.** Even inside Textual, marking a transcript block finalized lets you skip re-rendering it entirely. Keep `is_finalized()` and `settled_rows()` on the block base class, plus the stop-at-first-non-stable-child walk.
3. **Render-request coalescing with adaptive backpressure.** Do not call `widget.update()` per token: buffer deltas and flush on a ~30 Hz timer, and back off proportionally when the last paint was slow (omp's 50% duty-cycle rule, floored near 5 fps).
4. **Status line inside the input border** — pure layout, free to copy.
5. **One width model.** Pick `wcwidth` or `rich.cells.cell_len` and use it for *both* measuring and slicing. Never mix `len()` with cell width. omp's crash history is exactly this bug.
6. **Theme as a JSON token dict + adapter functions.** Keep `fg(token, text)` / `bg(token, text)` and per-subsystem adapters (`markdown_theme()`, `editor_theme()`) so subsystems never import the Theme class. Textual's `Theme` plus CSS variables map directly; keep the JSON files as source of truth and generate the CSS variables from them. Keep a `theme_epoch` counter to invalidate every render cache in one shot.
7. **Keybinding registry as action-id to chord-list, with a user YAML overlay** plus a canonical chord normalizer (modifier sort order, esc to escape, uppercase base implies shift, empty list disables). Textual's `BINDINGS` are static per widget, so add this indirection layer on top.
8. **Sync-vs-async autocomplete split.** Slash-command matching and inline expansion must be synchronous and I/O-free so Enter is deterministic; only path/file discovery goes async.

### Python MCP client options

| Option | Verdict |
|---|---|
| **`mcp` (official Python SDK, modelcontextprotocol/python-sdk)** | **Use this.** It ships `ClientSession` plus `stdio_client`, `streamablehttp_client`, and `sse_client` — the exact three transports omp hand-rolled. It also has `mcp.client.auth.OAuthClientProvider` with PKCE, RFC 9728 protected-resource discovery, RFC 8414 auth-server metadata, and RFC 7591 dynamic client registration, plus a `TokenStorage` protocol you implement. That is roughly 90% of `oauth-flow.ts` + `oauth-discovery.ts` for free. anyio/async-native, so it composes with Textual. |
| `fastmcp` (v2) | Ergonomic `Client` wrapper with multi-server config and transport auto-inference. Pleasant, but an extra layer over the same SDK; take it only if you want its config multiplexing out of the box. |
| Hand-rolled | No. omp did it because it predated a usable SDK and wanted Bun-specific spawn control. You have neither constraint. |

**What the SDK does NOT give you — port these from omp:**

- **The 250 ms fast-startup gate + deferred tool handles**, backed by a SQLite tool cache persisting `tools/list` results, so a slow server contributes cached callable tools immediately and the real connection is awaited lazily inside the call. Biggest perceived-latency win; no SDK equivalent. Python shape: `await asyncio.wait(connect_tasks, timeout=0.25)`, then build `DeferredTool` objects whose call path does `await manager.wait_for_connection(name)` first.
- **Reconnect + circuit breaker.** Backoff 0.5, 1, 2, 4 s on transport close; sliding 30 s window, more than 5 attempts suspends automatic reconnect; manual reconnect resets. Plus an epoch counter so a late reconnect cannot resurrect a connection after `disconnect_all()`.
- **Multi-source config discovery** (Claude, Cursor, VS Code, Codex, Gemini, Windsurf, OpenCode) with priority ordering and the *filter* vs *suppress* dedupe distinction. Cheap to port, high user-visible value.
- **Tool-name mangling and collision policy.** The `mcp__SERVER_TOOL` shape, sanitize to lowercase letters and underscores, strip a redundant server prefix, and pick a deterministic winner keyed on (server, original_tool) — never on iteration order.
- **Outbound arg hygiene**: strip harness-injected fields unless the server declares them, drop empty optional placeholders, resolve local URIs to real paths. Strict-schema servers reject calls otherwise.
- **Retriable-error classification with one reconnect + one retry**, and the tool-level `www_authenticate` challenge path (reauth, reconnect, retry once).
- **stdio process hardening.** POSIX: `start_new_session=True` (setsid) so a stdio server cannot be SIGTSTP'd by terminal job control — but **not on macOS**, where it breaks TCC / Apple Events attribution. Windows: the `.cmd` / `.bat` to `cmd.exe /c` BatBadBut escaping (CVE-2024-24576) is a real security control and Python's `subprocess` has the same hazard class.
- **Profile-scoped, URL-keyed credential ids** so a committed project `mcp.json` needs no `auth` block and each profile signs in as itself. Implement this as the key function of your `TokenStorage`.
- **Headless OAuth fallback**: print the full authorization URL hard-wrapped, and accept a pasted redirect URL or code, for SSH / WSL sessions where a loopback browser redirect cannot work.

### Suggested Python module layout

```
local_operator/
  tui/
    app.py             # Textual App: transcript / HUDs / input / status
    theme.py           # JSON token dict -> Textual Theme + CSS vars; fg()/bg() helpers; theme_epoch
    keys.py            # action-id registry + keybindings.yml overlay + canonical chord normalizer
    widgets/
      transcript.py    # finalized / settled-rows block protocol
      assistant.py     # streaming markdown with frozen prefix
      tool_card.py     # renderOutputBlock equivalent
      status_line.py   # rendered into the input border
      editor.py        # TextArea subclass + autocomplete provider chain
    autocomplete.py    # Provider protocol: sync slash + async file paths
  mcp/
    config.py          # multi-source discovery, filter/suppress, validation
    manager.py         # 250 ms connect gate, tool cache, reconnect + breaker, notifications
    tool_bridge.py     # name mangling, arg hygiene, result formatting, retry
    auth.py            # TokenStorage impl over the credential store
  agent/
    events.py          # AgentEvent union; TUI subscribes, mutates widgets, requests refresh
```

Keep the agent-to-UI boundary exactly as omp has it: **the agent emits typed events; the UI subscribes and owns all rendering.** That is what lets omp run one session under a TUI, RPC, ACP, and print mode without a rewrite, and it is the cheapest architectural property to preserve.

