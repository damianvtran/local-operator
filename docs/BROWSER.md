# Browser tool

The `browser` tool drives the **cmux embedded browser** — the browser panel
inside the terminal the operator is already running in. It is the only browser
backend, and it is advertised only when a cmux CLI can be reached.

Implementation: the `browser` section of `local_operator/tools/builtin.py`.
Tests: `tests/unit/tools/test_browser_tool.py`.

## Actions

| action | what it does | cmux command underneath |
|---|---|---|
| `open` | Start a surface at a URL (or navigate the one already open) | `--json new-surface --type browser --url <u> --focus false` |
| `goto` | Navigate the open surface | `browser --surface <s> goto <u>` |
| `read` | Page text, with the live title and URL | `browser --surface <s> get text --selector <sel>` |
| `snapshot` | Accessibility tree with `[ref=eN]` handles to click | `browser --surface <s> snapshot --compact` |
| `screenshot` | PNG to a path (verified before it is reported) | `browser --surface <s> screenshot --out <p>` |
| `click` | Click a CSS selector or a snapshot ref | `browser --surface <s> click --selector <sel>` |
| `type` | Replace the value of a field | `browser --surface <s> fill --selector <sel> --text <t>` |
| `close` | Close the surface and drop the handle | `close-surface --surface <s>` |

One surface per session, owned by the `Session` and injected into every rebuilt
`ToolContext` as `ToolContext.browser`. `open` called a second time navigates
that surface instead of creating another, and `Session.dispose()` closes
whatever is still open. The holder has to be host-owned because
`Session._run_turn` rebuilds the `ToolContext` at the start of *every* turn: a
handle the tool stored on the context it was handed survived exactly one turn,
which broke `open X` → next message → `click Y` outright and stranded a cmux
tab per turn that nothing could close.

## Detection

`cmux_browser_available()` is a **PATH lookup and an environment read, with no
subprocess**. It runs while the tool inventory is being built on every session
start, and session start must never block on a terminal emulator or a wedged
socket.

Resolution order in `_cmux_binary()`:

1. `shutil.which("cmux")`.
2. `$CMUX_BUNDLED_CLI_PATH`, if it points at an executable file. A cmux session
   exports this pointing into the app bundle
   (`/Applications/cmux.app/Contents/Resources/bin/cmux`). It matters because
   cmux's shell integration *prepends* that bin directory to `PATH`, and a venv
   activation, a `sudo -i`, or a login shell that rebuilds `PATH` from
   `/etc/paths` drops it while every `CMUX_*` marker survives.

**The binary is the gate; environment markers alone are not enough.** `CMUX_*`
is inherited by every descendant of a cmux session, including ones that crossed
into a container or an ssh host where no cmux CLI exists. Detecting on a marker
there advertised a tool whose every action could only answer *"cmux is not on
PATH"*.

Signals measured inside a real cmux session on this host (2026-08-06):

| variable | value | usable as a signal? |
|---|---|---|
| `CMUX_SOCKET` | **empty string** | No — always falsy. The previous `os.environ.get("CMUX_SOCKET")` check could never fire. |
| `CMUX_SOCKET_PATH` | `~/.local/state/cmux/cmux-501.sock` | Populated, but proves nothing about a CLI being present. |
| `CMUX_SURFACE_ID`, `CMUX_PANEL_ID`, `CMUX_WORKSPACE_ID` | UUIDs | Inherited by every child process; not a capability. |
| `CMUX_BUNDLED_CLI_PATH` | path into the app bundle | **Yes** — an executable path, checked as the PATH fallback. |
| `which cmux` | `/opt/homebrew/bin/cmux` | **Yes** — the primary signal. |

Degrading is silent to the MODEL. No cmux means `build_browser_tool()` returns
`None` and the tool is never advertised (the *createIf* convention, same as
`wake`). Nothing raises, and session start is unaffected. If a host forces the
tool on anyway, every action returns one clear error.

It is not silent to the LOG. A session carrying cmux's `CMUX_*` markers — so
plainly running inside cmux — that still resolves no CLI is the one anomalous
shape, and `cmux_browser_available()` logs a warning naming the markers it saw
and the `CMUX_BUNDLED_CLI_PATH` it rejected. A `PATH` rebuilt by a login shell
or a `sudo -i` is exactly how that happens. An ordinary non-cmux host logs
nothing: absence there is normal, and a warning per session start is noise that
trains everyone to ignore the real one.

**There is no headless fallback, deliberately.** This repo ships no browser
engine: `playwright` belongs to the pre-rewrite codebase, appears in no
dependency group in `pyproject.toml`, and is not installed in the venv. Adding
one would put ~10 packages and a ~150 MB browser download into a default
install that is kept small on purpose. A host without cmux therefore has no
browser tool at all — which is honest — and the agent still reaches static
pages with `bash` and `curl`.

## The substitution gap, and the three places it is closed

Detection being right is not enough. Measured in a real session (transcript
`d85fe6050bc0`, 2026-08-17), with `cmux_browser_available()` true and `browser`
in the inventory: asked for before/after screenshots of a local dev server, the
agent wrote a `capture-my-work-evidence.mjs` playwright script and ran
`pnpm exec playwright install chromium` — 23 s of browser download. Told
outright to *"use the cmux browser instead"*, it read its cmux guidance and then
shelled `cmux --json new-surface` through `bash`, still never calling the tool.
The tool was reachable the whole time; nothing told the model what it was for.

A downloaded headless Chromium is not a smaller version of the real thing. It
carries none of the user's cookies, the user cannot log into it, and it renders
where nobody can look — so for anything behind a login it is a dead end that
looks like progress. That property, not the verb list, is the reason to pick
this tool, so it is now stated in all three places the model can read:

1. **The tool description** says it drives the user's REAL browser, that
   cookies and logins persist across calls and sessions, that the user can be
   asked to sign in by hand, and that a browser engine must never be installed
   instead.
2. **The system prompt** (`prompts_md/system.md`) carries the general rule: no
   `playwright install`, no puppeteer, no downloaded Chromium — and, when no
   browser tool is listed, say screenshots are unavailable and why rather than
   building a parallel browser stack.
3. **The tool inventory** (`prompts_api._NO_BROWSER_NOTE`) names the absence
   when there is no `browser` tool, because an unexplained hole reads as
   "arrange your own". It is keyed on membership rather than visibility: a
   hidden tool still answers, and claiming otherwise would be worse than
   silence. The `createIf` contract is untouched — a tool that cannot work is
   still never advertised.

Personal cmux workflow rules (which surface to reuse, focus habits, layout)
belong in the operator's own `system_prompt.md`, not in this package.

## cmux conventions, and why they are not negotiable

The operator runs one agent session per cmux workspace: a single terminal
surface in a single pane, hand-arranged.

- **Open with `--json new-surface --type browser --url <u> --focus false`.**
  This adds the browser as a sibling **tab** in the calling pane.
- **Never `cmux browser open`, `open-split`, or `new`.** They reuse a right-hand
  pane if one exists and otherwise **split the pane in two**. Nothing heals
  that; the operator rebuilds the layout by hand.
- **Never pass `--workspace` or `--pane`.** No `--pane` is what makes the
  socket resolve the calling terminal's own pane, which is the pane the browser
  should join. Omitting `--workspace` does **not** likewise avoid
  `$CMUX_WORKSPACE_ID` — `cmux new-surface --help` documents
  `--workspace <id|ref|index>  Target workspace (default: $CMUX_WORKSPACE_ID)`,
  so cmux applies that default server-side either way. It is omitted because
  passing it explicitly can only make things worse (a value we compute from a
  stale env var, against cmux resolving its own current one).
- **`--focus false` when CREATING a surface** — that is the only place the tool
  passes it, and the only place cmux accepts it: `cmux browser --help` lists
  `--focus` on `open|open-split|new` alone, and `cmux new-surface --help` on
  the creation call. `goto`, `click`, `snapshot` and friends take no `--focus`,
  so do not add one when adding an action. cmux only activates on an explicitly
  truthy focus, so an unfocused creation never raises the window over what the
  user is doing.
- **One surface, reused, then closed.** A fresh surface per navigation leaves a
  drift of dead tabs the operator closes one at a time.

Verified after the live run below — one pane, two terminal surfaces, no browser
tab left behind, no split:

```
├── workspace workspace:7 ".venv/bin/python -m local_operator.cli"
│   └── pane pane:7 [focused]
│       ├── surface surface:10 [terminal] …
│       └── surface surface:32 [terminal] …
```

## Five cmux behaviours the tool has to defend against

All five were measured against the real CLI; each has a regression test.

### 1. `get url` is the URL cmux was *asked* for, not the one that is loaded

`goto` exits 0 the instant the request is accepted. After
`goto https://iana.org/domains/example` — a 301 this WKWebView never completes —
`cmux browser get url` reported the requested URL for **20+ seconds** while
`location.href`, `get title` and `screenshot` all still described the *previous*
page. The screenshot was byte-identical to the earlier one
(md5 `cef9cd9d088372002bb428604ffb43d5`, 67 821 B). No exit code says so.

So every navigating action waits for the two views to agree
(`_await_navigation`): settled means `readyState == "complete"` **and** cmux's
URL equals the document's `location.href`. It is redirect-safe because both
sides report post-redirect state — `www.rust-lang.org/learn` settles with both
reading `https://rust-lang.org/learn/`. A navigation that never lands is an
error naming both URLs, never a success.

### 2. A click needs two different signals to prove it navigated

A `goto` updates cmux's URL synchronously; a click is initiated by the *page*,
so for a moment both readings still agree on the old URL and the settle
predicate would call that "already settled".

- Link click → cmux's URL flips within the first poll. Measured.
- **Form POST to the same URL** → no URL changes at all. Measured against
  DuckDuckGo's no-JS search form: a marker property set on the document
  (`window.__lo_nav`) cleared ~0.6 s after submit while the URL never moved.

`_navigation_started` waits up to `BROWSER_CLICK_GRACE_S` (1.5 s) for *either*
signal, then either settles the navigation or reports `(no navigation)`.

### 3. `goto` is an omnibox

`cmux browser goto 'not a url at all'` exits 0 with `OK` after landing on
`https://www.google.com/search?q=not%20a%20url%20at%20all`. A `data:` URL is
search-escaped the same way. A typo'd or hallucinated URL would therefore
produce a search-results page that every later read and screenshot describes as
if it were the requested site, so only `http://` and `https://` are accepted,
and the refusal happens before the subprocess runs.

### 4. A dead `--surface` handle silently drives the *active* surface

cmux resolves a handle that no longer exists by falling back to whatever
surface is active, and exits 0. Measured against `--surface surface:999999`, a
handle that never existed:

| command | rc | result |
|---|---|---|
| `get url` | **1** | `Error: invalid_params: Missing or invalid surface_id` |
| `get title` | 0 | an unrelated tab's title |
| `get text --selector body` | 0 | that tab's full text |
| `eval --script …` | 0 | that tab's document |
| `snapshot --compact` | 0 | that tab's tree |

Driven through the tool with a stale handle, `read` answered `is_error: False`
with a confident page header and that page's whole body while
`details.surface_id` still named the dead handle — internally consistent and
completely wrong, with nothing in the transcript signalling the substitution.
`_await_navigation` cannot catch it either: its `eval` probe *succeeds* against
the fallback surface, so `probe_failures` never reaches 3.

Handles go stale routinely — the user closes the tab, or cmux restarts and
reissues small refs like `surface:73` to someone else.

`get url` is the only usable liveness probe *because* it is the one verb that
refuses. `execute_browser` runs exactly one before dispatching any
surface-taking action (`_stale_surface_error`) — never inside a poll loop — and
on failure drops the handle and tells the model to `open` again. `open` itself
recovers instead of erroring: it is the verb that would fix this anyway.

### 5. `fill` exits 0 without filling

cmux's parser is flag-greedy in the `--text` slot:
`fill --selector a --text --help` exits 0 and prints the browser help. Ordinary
dash-leading values (`-5`, `-x`, `--force`) do fill correctly, but the tool
refuses any leading dash anyway — same rule as the selector and the URL, because
the alternative is an allowlist of another program's global flags.

`type` then **compares** the `get value` read-back against what was asked for
and reports a mismatch as an error. It used to interpolate the read-back into
`Value is now 'X'.` without comparing, so a fill that did nothing was reported
as a success quoting the field's OLD contents as the new ones. A target with no
`value` property (a contenteditable) says `unverified` rather than claiming a
confirmation it does not have.

Two smaller ones:

- **`get text` is `innerText`, which needs layout.** A surface in a background
  tab may never lay out. On a real results page both `get text --selector body`
  and `document.body.innerText` returned `""` while `textContent` held 15 247
  characters. `read` falls back to a DOM walk (script/style stripped,
  whitespace collapsed) rather than reporting `(no text)`.

  The same `innerText` definition has a second edge worth knowing: on a subtree
  that is *never* rendered it returns `textContent` instead, style and script
  bytes included. Measured against `example.com` with `selector: head` —
  `get text` returned `Example Domainbody{background:#eee;width:60vw;…}`, the
  page's inline CSS glued to its title, while the DOM walk returned exactly
  `Example Domain`. The fallback fires only on an *empty* result, so reading a
  non-rendered subtree can still surface CSS. That is left alone deliberately:
  the alternative is a heuristic that guesses when real page text "looks like"
  a stylesheet, and a wrong guess silently deletes content. Normal `read`
  (default `body`) is unaffected.
- **`screenshot` needs `--out`.** Passed positionally, cmux ignores it, writes
  into its own temp dir and still exits 0. The tool also checks the file exists
  and starts with the PNG magic bytes before telling the model it can read it.

## Live evidence

Driven through the registry-built tool (`create_tools`) against real cmux and
real pages, 2026-08-06. Full transcript of the run:

```
tools advertised: ['bash','browser','edit','glob','grep','list_variables',
                   'read','read_variable','todo','write']

open  https://example.com   -> Opened browser surface surface:90:
                               Example Domain — https://example.com/
read                        -> "Example Domain / This domain is for use in
                               documentation examples without needing
                               permission. Avoid use in operations. / Learn more"
snapshot                    -> - document "Example Domain"
                                 - heading "Example Domain" [ref=e14]
                                   - link "Learn more" [ref=e15]
screenshot /tmp/lo-live/example.png
                            -> 67821 bytes
click a                     -> ERROR: clicked a, but the page it started loading
                               never arrived: after 20s cmux is pointing at
                               https://iana.org/domains/example but the live
                               document is still https://example.com/
goto  https://html.duckduckgo.com/html/
                            -> DuckDuckGo HTML: Private Search Without JavaScript
type  input[name=q] "local-operator harness"
                            -> Value is now 'local-operator harness'
click input[type=submit]    -> Clicked. Page: local-operator harness at
                               DuckDuckGo — https://html.duckduckgo.com/html/
read  body                  -> "GitHub - damianvtran/local-operator: AI agents
                               platform … github.com/damianvtran/local-operator"
screenshot /tmp/lo-live/ddg.png
                            -> 261218 bytes
goto  "not a url at all"    -> ERROR: refusing 'not a url at all': only http://
                               and https:// can be opened
goto  "--help"              -> ERROR: refusing a flag-shaped URL: '--help'
close                       -> Closed browser surface surface:90.
                               surface handle after close: ''
```

Screenshot verification (magic bytes read back off disk):

| file | bytes | first 8 | md5 |
|---|---|---|---|
| `/tmp/lo-live/example.png` | 67 821 | `\x89PNG\r\n\x1a\n` | `cef9cd9d088372002bb428604ffb43d5` |
| `/tmp/lo-live/ddg.png` | 261 218 | `\x89PNG\r\n\x1a\n` | `a3bd32e28526c96c55946aec6ea25f10` |

The two differ, which is what shows the capture follows the live page rather
than replaying a cached frame.

Second live run, isolating the DOM fallback (the first run never triggered it —
cmux's `get text` worked on every page it visited). `head` is never rendered,
which makes the path reproducible on demand:

```
open https://example.com                        -> surface:92
raw `get text --selector head`                  -> 'Example Domainbody{background:#eee;
                                                    width:60vw;…a:link,a:visited{color:#348}'
raw DOM walk (_dom_text_js("head"))             -> 'Example Domain'
close                                           -> Closed browser surface surface:92.
```

That is the fallback's extraction proven against a live page: it drops the
inline stylesheet cmux's own text extraction includes.

The `click a` error is the tool working, not failing: `iana.org/domains/example`
301s to `www.iana.org`, and this WKWebView never completes it — reproduced four
times, twice through the raw CLI. Before the settle check, that click reported
success and the next read and screenshot silently described example.com.

Cleanup after the run: `cmux tree` showed no browser surface in the workspace,
`pgrep -fl cmux` showed only the cmux app itself, and no Chrome process was
started — the tool drives cmux's WKWebView and never spawns a browser.
