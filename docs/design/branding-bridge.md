# Branding brief: the browser extension + local bridge

Working brief for naming and copy. The product: a free Chromium extension plus a small
local bridge service that lets a Local Operator agent drive the user's own browser —
navigate, read, click, type, screenshot — with the user's existing logins, entirely
on-device. Per-site allow list and a pairing code gate it. Published free on the Chrome
Web Store, supported by Radient, Inc.

Constraint: competitor prior art is **Kimi WebBridge**. "WebBridge", "web bridge", and
"browser bridge" are off the table as names. Descriptive use in body copy ("a small
local bridge") is fine.

Brand voice (from local-operator.com): ordinary words given a job. Short declaratives.
Benefit first, no hype. "Local Operator" is itself a telephone-operator metaphor, and
"Radient Pass" follows the same pattern — a plain word doing a specific job.

## 1. Candidates

**Switchboard-heritage family** (the operator's own tools):

1. **Patch** — what a telephone operator does: patch a call through. The agent gets
   patched into your browser. Short, ordinary, verbs well ("patch the agent in").
2. **Patchcord** — the physical cable the operator plugs in. More distinctive than
   Patch, but reads techy/audio-gear and is two ideas where one will do.
3. **Line** — "the operator has a line to your browser." Dead ordinary, but too
   generic to search for and collides with the LINE messenger extensions.
4. **Exchange** — the building the operator sits in. Wrong direction: the extension
   is a cord, not a headquarters, and "Exchange" is owned by Microsoft in most minds.
5. **Extension** itself is prior art here — a telephone extension is literally a second
   handset on the same line, which is exactly what this is. Ungoogleable as a name, but
   worth stealing for copy: "a second handset on your line."

**Hands/eyes/reach family** (the agent gets into the browser):

6. **Handset** — the part of the phone you actually hold. The agent picks up your
   browser the way you would. Warm, ordinary, unclaimed in the store.
7. **Reach** — the agent's reach extends into the browser. Verb-noun, on-voice, but
   abstract, and the store is crowded with growth-marketing "Reach" tools.
8. **Window** — the browser is the agent's window. Too generic; collides with the OS word.

**Plain-compound family**:

9. **Sidecar** — rides along with your browser. Good metaphor, but Apple's Sidecar
   (screen extension!) owns the word and several extensions already use it.
10. **Local Operator for Chrome** — the no-name name. Safe, but wastes the chance to
    name the capability, and lies by omission on Edge/Arc/Brave.

## 2. Collision screen (top 3)

| Name | Chrome Web Store | Web / trademark | Verdict |
|---|---|---|---|
| **Patch** | "Page Patch" (element remover), "Web Patch" (unit converter), "Prompt Patch" (prompt tool) — all small, unrelated categories; no AI-browser-agent named Patch | Red Hat "patch-operator" is a Kubernetes tool, different market; no extension trademark found | **Clear.** Store listing will read "Local Operator Patch", which shares no confusing overlap with any of these |
| **Handset** | No extension named Handset found | Generic telephony word; no software claimant surfaced | **Clear**, though weaker search presence to defend |
| **Reach** | No direct agent-extension hit, but many "Reach"-branded outreach/sales tools adjacent to browser automation | Crowded; multiple marketing SaaS marks named Reach | **Risky** — avoid |

Also checked: "Switchboard" (three-plus existing extensions, one an AI-provider bridge —
burned), "Sidecar" (Apple + existing extensions — burned).

## 3. Recommendation

**Winner: Patch** (store name **Local Operator Patch**, UI name **Patch**).

It is the switchboard metaphor completing itself. Local Operator is the operator;
Patch is how the operator patches into your browser. One syllable, an ordinary word
given a job — exactly the "Radient Pass" pattern. It verbs naturally in copy and UI
("Patched in", "Patch into this site?"), which none of the other candidates do. The
store collisions are small tools in unrelated categories, and the compound
"Local Operator Patch" is unambiguous.

**Runner-up: Handset.** Same heritage, warmer, and clear of collisions — but it
describes holding the phone rather than connecting the call, and it doesn't verb.
If Patch fails a legal check, Handset is ready.

## 4. Copy kit (Patch)

### Store listing

**Title** (20/45 chars):
> Local Operator Patch

**Short description** (116/132 chars):
> Patch your Local Operator agent into the browser you already use. Your logins stay put. Nothing leaves your machine.

**Long description:**

> **Your browser, with an agent patched in.**
>
> Patch connects the free, open-source Local Operator app to the browser you already
> use — Chrome, Edge, Arc, or Brave. The agent can open pages, read them, click,
> type, and take screenshots, in your real browser with your real logins. Sign in to
> a site by hand and the agent carries on from there.
>
> **Everything stays on your machine.**
>
> Patch talks to a small bridge service running on your own computer. No cloud
> browser, no remote session, no page content sent anywhere. What happens on your
> machine stays on it.
>
> **You decide which sites.**
>
> The agent can only use sites you've allowed. The first time it wants a new one,
> Patch asks, and you answer once. A pairing code ties the extension to your own
> Local Operator app and nothing else.
>
> **Free, like the rest.**
>
> Patch is free and pairs with the free, MIT-licensed Local Operator app.
> Supported by Radient, Inc.

### Extension popup microcopy

Connection states:
- Connected: **"Patched in."** — sub: "Local Operator can use this browser."
- Disconnected: **"Not connected."** — sub: "Open Local Operator to reconnect." button: "Retry"
- Pairing: **"Waiting to pair."** — sub: "Enter the code shown in Local Operator."

Pairing prompt:
- Heading: **"Pair with Local Operator"**
- Body: "Type the 6-digit code from the Local Operator app. Pairing links this
  browser to your app and nothing else."
- Field label: "Pairing code" · Button: "Pair" · Error: "That code didn't match. Codes expire after two minutes — check the app for a fresh one."

Per-site allow prompt:
- Heading: **"Let the agent use github.com?"**
- Body: "The agent wants to open a page on this site. It will use your current
  login. You can take this back any time in Settings."
- Buttons: **"Allow"** / **"Not this site"** · Checkbox: "Just this once"

Settings labels:
- "Allowed sites" (list; per-row "Remove")
- "Ask before each new site" (toggle, default on)
- "Show the agent's cursor while it works" (toggle)
- "Unpair this browser" (button, destructive)

### README / website blurb

> Patch puts your Local Operator agent in your real browser. It navigates, reads,
> clicks, types, and takes screenshots — with your existing logins, entirely on your
> machine. A per-site allow list and a pairing code keep it yours: the agent uses only
> the sites you've approved, and only from your own copy of the app. Free on the
> Chrome Web Store, works in any Chromium browser, supported by Radient, Inc.

### Taglines

1. "Your browser, patched in."
2. "The operator has a line to your browser."
3. "Real browser. Real logins. Your machine."
4. "Sign in once. The agent carries on."
5. "It browses where you already are."

## 5. Code artifact naming

- Python package: `local_operator/patch/` (bridge service; module docstring explains
  the telephone-operator metaphor so the name reads as intent, not a diff patch)
- Extension dir: `extensions/patch/` (repo) → store artifact `local-operator-patch`
- CLI noun: `lop patch ...` — `lop patch status`, `lop patch pair`, `lop patch allow <site>`
- Docs: `docs/patch.md`

One caveat to record: inside a codebase, "patch" also means a diff. The CLI surface
(`lop patch pair`) is unambiguous in context, but code comments and docs should say
"the Patch bridge" on first mention, never bare "patch", to keep grep and prose clear.
