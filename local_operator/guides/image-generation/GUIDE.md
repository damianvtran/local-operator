---
name: image-generation
description: Generate or edit images via the configured provider — setup, edit-model advice, cost, cancel and restart, and what to do when no provider is configured.
---

# Generating and editing images

`generate_image` turns a prompt into an image (or, with a source —
`source_image_path` or `source_attachment` — edits one) and attaches the
result to the session. You do not save files or
move bytes around: the harness registers each image in the session's
attachment store, surfaces render it, and the caption carries the digests.

## Which provider runs

The harness walks providers in a fixed, append-only order and uses the
FIRST one with a credential. **Failover is automatic**: if a provider
refuses (bad key, no credits, model trouble), the next one runs, and the
result's `details.attempts` records what each said.

| Order | Provider | What it needs |
|---|---|---|
| 1 | Radient | a signed-in account (`/login radient`) |
| 2 | FAL | a stored key (`lop login fal`, or an exported `FAL_API_KEY`) |
| 3 | OpenAI | a platform API key (`lop login openai-key`, or an exported `OPENAI_API_KEY`) |
| 4 | ChatGPT plan | the ChatGPT subscription sign-in (`lop login openai`) |
| 5 | Google | an AI Studio key (`lop login google`, or an exported `GOOGLE_AI_STUDIO_API_KEY`) |
| 6 | xAI | a key (`lop login xai`) or the Grok sign-in (`lop login xai-oauth`) |
| 7 | OpenRouter | a key (`lop login openrouter`, or an exported `OPENROUTER_API_KEY`) |

There is no knob to pick a provider for one call — the order is the rule.
Choosing `model` narrows what the winning provider runs, not who runs.

Rungs 4–7 are append-only: they sit AFTER the original three, so an existing
setup's path never changes. The **ChatGPT-plan rung is a subscription rung** —
it spends plan quota instead of a billed key, and it runs only when the
earlier rungs refuse or are absent (with a platform key stored, the key rung
wins: it supports seeds the subscription route cannot).

Provider-by-provider provenance, wire shapes and verified-at dates:
`docs/design/image-providers.md`.

## Setup

- **Radient**: `/login radient` (or `lop login radient` from a shell).
- **FAL**: `lop login fal` — paste a key from your FAL dashboard. An exported
  `FAL_API_KEY` also works.
- **OpenAI**: `lop login openai-key` — paste a platform API key. An exported
  `OPENAI_API_KEY` also works.
- **ChatGPT plan**: `lop login openai` — the ChatGPT subscription sign-in the
  chat side already uses. Subscription logins CAN fund images through the
  Codex backend's built-in image tool: that burns quota 3–5× faster than text
  turns, has no seed, is not available on the Free plan, and the images API
  itself still needs a platform key. Expect 1–4 minutes per image.
- **Google**: `lop login google` — paste an AI Studio key. An exported
  `GOOGLE_AI_STUDIO_API_KEY` also works. Runs the Nano Banana family; Imagen
  is shut down in this API.
- **xAI**: `lop login xai` (API key) or `lop login xai-oauth` (the Grok
  sign-in). xAI tiers its OAuth surface — a subscription sign-in can be
  refused with an HTTP 403 even while active, so the **stored key is used
  first and the Grok sign-in is the fallback** (an exported `XAI_API_KEY`
  counts as a key and is honoured last).
- **OpenRouter**: `lop login openrouter` — one key fronts most of the public
  image catalogue. An exported `OPENROUTER_API_KEY` also works.

Keys are stored in the encrypted store and never printed. `guide://credentials`
covers the general rules for handling them.

## Cost

- **Radient**, **xAI** and **OpenRouter** report a per-generation cost — it
  appears as `cost_usd` in the result (`usage.cost_in_usd_ticks` on xAI,
  `usage.cost` on OpenRouter). Radient also bills account credits and is
  skipped when the balance cannot fund the request.
- **FAL**, **OpenAI** and **Google** bill the key you supplied at their
  published rates; **the ChatGPT-plan rung spends plan quota instead** — no
  cash charge, and the 3–5× burn above is the price. Rate tables with
  provenance and dates: `docs/design/image-providers.md`; prices are not
  duplicated into code except the labelled constants: the ChatGPT-plan
  API-equivalent, and the GPT-image per-token rates the OpenAI edit estimate
  multiplies (every multiplier there is provider-reported usage).
- Every figure carries a `billing_basis` (and `cost_source`, `cost_provenance`)
  in the result details: `billed` (cash/credits charged — Radient's settled
  figure, xAI on an API key, OpenRouter), `estimated` (a figure that may still
  be a quote — Radient when the status never reports `settled` — or an OpenAI
  edit computed from the response's reported token counts at published
  rates), or `subscription-api-equivalent` (the **ChatGPT-plan** rung: ≈ $0.053
  per image, OpenAI's published `gpt-image-2` 1024×1024 medium price, fetched
  2026-10-09 — an assumption, since the route pins no size/quality; and xAI on
  a Grok sign-in, using xAI's reported figure). An API-equivalent is what the
  same call would list at on the API — **never a charge**.
- **Edits can price differently from same-model generations** (xAI bills the
  input image *and* the output image; OpenAI is token-priced). The result
  carries only what the provider reports — an OpenAI edit estimates from its
  reported usage tokens and the published rates, and everything else stays
  figure-free rather than guessing.
- `num_images` multiplies cost on every rung — it is the spend knob, and the
  approval prompt states the quantity before anything is spent. xAI accepts
  up to 10; OpenRouter models may return fewer than asked; the ChatGPT-plan
  and Google rungs generate one image per call (a larger request fails over
  rather than silently delivering fewer).

The harness asks for approval before spending (the tool is write-tier in the
default "ask" mode). In unattended modes there is no prompt — treat
`generate_image` as a paid call wherever it appears.

## Model choice

`model` takes a provider-specific id and is optional everywhere:

- **Radient**: any id from its live model list (read at call time).
- **FAL**: an app path like `fal-ai/flux/dev` (the default).
- **OpenAI**: a `gpt-image`-class id (a default runs when omitted).
- **ChatGPT plan**: the current Codex host model runs by default; override
  only if you know the catalogue.
- **Google**: `gemini-nano-banana-2.1` (default) or another Nano Banana
  family id.
- **xAI**: `grok-imagine-image-2.0` (default).
- **OpenRouter**: any image model slug from its catalogue — e.g.
  `bytedance-seed/seedream-4.5` (default),
  `black-forest-labs/flux.2-klein-4b` (cheap), or `openai/gpt-image-2.5-*`
  (flagship).

`image_size` accepts `square_hd`, `square`, `portrait_4_3`, `portrait_16_9`,
`landscape_4_3`, `landscape_16_9` and is mapped per provider. `seed` runs only
where the provider supports it: Radient, FAL and OpenRouter (per model);
OpenAI, Google and the ChatGPT-plan route cannot honour it — do not promise
reproducibility there.

## Editing an image (image-to-image)

Pass one or more sources and the call becomes an edit. Four fields name
them:

- `source_image_path` / `source_image_paths` — local files; read locally and
  uploaded to the provider as data URIs (one path, or a list, in order).
- `source_attachment` / `source_attachments` — digests from previous results'
  captions (`generate_image` captions name the digest of every image they
  attached); pass one back, or a list, to edit images the harness just made.

Kinds mix freely in one call: the combined order is every path first, then
every attachment, each list in the order given (a two-path + one-digest call
edits path1, path2, digest1). One source is the classic single-edit; two to
sixteen sources combine in one request — 16 is the hub's highest per-pin
ceiling and the tool's per-call maximum (`At most 16 sources per call`).
The plural fields are alternatives to their singular twins, not companions:
pass `source_image_path` or `source_image_paths`, not both (same for the
attachment pair).

Per-provider ceilings decide what actually runs. Radient reads the ceiling
off the live listing per pin at call time (its edit default, GPT Image 2.5
sunburst edit, takes 16); FAL's multi-reference editors take up to 10
(`blackforestlabs/flux-3/edit-image`, the unpinned multi-source default — a
pinned single-reference app is skipped, never swapped); OpenAI's GPT-image
models take up to 16; Google up to 14; xAI up to 5; OpenRouter takes one
reference in this version. A provider that cannot take the count is skipped
with its ceiling named, and the ChatGPT-plan rung skips edits explicitly —
an edit never silently becomes a fresh image or a quietly smaller edit.

Each source rides as a data URI or an https URL; the Radient hub refuses a
request body over 16 MiB, so many large local sources can hit that cap
before the source-count cap does.

`strength` (0..1) tunes how far the edit may move from the source(s), where
the provider supports it (FAL's flux-class image-to-image apps; its
multi-reference editors document no strength field). Where the harness can
detect the drop — FAL's schema-guided path — the result says the value was
ignored rather than dropping it silently; on Radient it rides the hub's
verbatim passthrough, and a pin that does not document it gives no drop
signal, so the result does not claim one. `strength` requires at least one
source.

Edits run on Radient, FAL, OpenAI, Google, xAI and OpenRouter. Cost: FAL and
Google report no per-call figure — their captions say so explicitly — and
bill at their published rates on your key; an OpenAI edit shows an estimated
figure when the response carries usage, and otherwise says no figure was
reported; Radient, xAI and OpenRouter report per-call figures as before.

**Which edit model.** On Radient the harness resolves the pin the hub marks
as its edit default — GPT Image 2.5 sunburst edit, the best single-edit
quality on the listing. It **drifts across successive edits** (each round is
free to move the whole image): for multi-round revision workflows prefer an
NB2.1-class or flux-class model via FAL or OpenRouter. Radient value pins:
Nano Banana 2 edit ≈$0.08 and Nano Banana Pro edit ≈$0.15 per image;
premium: FLUX.2 max edit. On every provider, pin `model` to choose
specifically.

Digests come from results. An image pasted straight into the chat has no
digest you can cite yet — reference a local file with `source_image_path`
(or `source_image_paths`) when the bytes exist on disk.

## Cancelling and restarting

Stopping the turn (Esc, `/stop`, a steering message) stops waiting on the
generation and best-effort-cancels the provider-side job. **A cancelled
generation is not resumed**: ask again (the receipt and caption carry the
prompt, model and seed, so a re-issue can repeat or edit it).

Restarting the harness changes nothing about this: every generation is a
fresh call with a fresh provider job. There is no job to reconnect to.

## When nothing is configured

The guide is readable with no provider set up — that is the moment it is for.
Pick one of the setup paths above, then generate: the tool appears in sessions
built after a credential lands.

If a call reports every provider failed, read the attempts in the error note:
they name each provider's refusal (a refused key, exhausted credits, a model
that does not exist). Fix the named thing and retry.

## MCP image tools

If this session already holds MCP tools that generate images, call them
directly for their providers — `generate_image` never proxies MCP tools, and
it does not know about them.

## Naming the result

`filename` sets the attachment's display name — the Library row, the caption
label and the session-file entry — from a short lowercase hyphenated stem, no
extension:

```json
{"prompt":"A red fox in snow, painterly","filename":"red-fox-in-snow"}
```

Omitted, the stem derives from the prompt (`A red fox in snow, painterly` →
`red-fox-in-snow-painterly`). The extension always follows the bytes the
provider returned (`image/webp` → `.webp`), never the stem. A stem is a label,
not a path: separators and `..` are sanitised away, and the file materialises
under the session's `scratchpad/generated/`; two different images with one
name never overwrite — the second FILE gets a short digest suffix, while its
Library row and caption keep the requested label.

## Saving to disk

Attachments ARE the delivery: the image is rendered by the user's surfaces
from the session store, and the caption's digest is the handle. Save a file to
disk only when the user asks for one (an ordinary `bash` command, from the
attachment digests the result names).
