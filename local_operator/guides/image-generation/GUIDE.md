---
name: image-generation
description: Generate or edit images with Radient, FAL, or OpenAI — provider setup, credits, model choice, cost, cancel and restart, and what to do when no provider is configured.
---

# Generating and editing images

`generate_image` turns a prompt into an image (or, with `source_image_path`,
edits one) and attaches the result to the session. You do not save files or
move bytes around: the harness registers each image in the session's
attachment store, surfaces render it, and the caption carries the digests.

## Which provider runs

The harness walks providers in a fixed order and uses the FIRST one with a
credential. **Failover is automatic**: if a provider refuses (bad key, no
credits, model trouble), the next one runs, and the result's
`details.attempts` records what each said.

| Order | Provider | What it needs |
|---|---|---|
| 1 | Radient | a signed-in account (`/login radient`) |
| 2 | FAL | a stored key (`lop login fal`, or an exported `FAL_API_KEY`) |
| 3 | OpenAI | a stored API key (`lop login openai-key`, or an exported `OPENAI_API_KEY`) |

There is no knob to pick a provider for one call — the order is the rule.
Choosing `model` narrows what the winning provider runs, not who runs.

## Setup

- **Radient**: `/login radient` (or `lop login radient` from a shell).
- **FAL**: `lop login fal` — paste a key from your FAL dashboard. An exported
  `FAL_API_KEY` also works.
- **OpenAI**: `lop login openai-key` — paste a platform API key. An exported
  `OPENAI_API_KEY` also works. A ChatGPT subscription login does NOT run image
  calls; the images API needs a platform key.

Keys are stored in the encrypted store and never printed. `guide://credentials`
covers the general rules for handling them.

## Cost

- **Radient** reports the generation's cost (`cost_usd` in the result) and
  bills your account credits; when the balance cannot fund the request the
  rung is skipped and the next provider runs.
- **FAL** and **OpenAI** bill the key you supplied; OpenAI bills **per
  image**. `num_images` multiplies cost on every provider — it is the spend
  knob, and the approval prompt states the quantity before anything is spent.

The harness asks for approval before spending (the tool is write-tier in the
default "ask" mode). In unattended modes there is no prompt — treat
`generate_image` as a paid call wherever it appears.

## Model choice

`model` takes a provider-specific id and is optional everywhere:

- **Radient**: any id from its live model list (the list is read at call
  time; its default image model runs when you omit `model`).
- **FAL**: an app path like `fal-ai/flux/dev` (the default).
- **OpenAI**: a `gpt-image-1`-class id (the default) or a `dall-e-3`-class id.

`image_size` accepts `square_hd`, `square`, `portrait_4_3`, `portrait_16_9`,
`landscape_4_3`, `landscape_16_9` and is mapped per provider. `seed` runs only
where the provider supports it (Radient and FAL; OpenAI ignores it — do not
promise reproducibility on that rung).

## Editing an image (image-to-image)

Pass `source_image_path` (a local file) and the call becomes an edit: the file
is read locally and uploaded to the provider as a data URI. `strength` (0..1)
tunes how far the edit may move from the source, where the provider supports
it. Without `source_image_path`, `strength` is rejected.

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

## Saving to disk

Attachments ARE the delivery: the image is rendered by the user's surfaces
from the session store, and the caption's digest is the handle. Save a file to
disk only when the user asks for one (an ordinary `bash` command, from the
attachment digests the result names).
