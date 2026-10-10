---
name: browser-qa
description: "Testing web apps through a real browser: reconnaissance before acting, selector discovery, state capture, console errors and screenshot evidence. Use on a web surface end to end, or to reproduce a front-end bug."
---

# Browser QA

Exercise the web surface with the harness `browser` tool: reconnaissance, then action, then capture the resulting state. Concrete flows: `references/recipes.md`.

## Recon before acting

1. Open the target and wait for READY, not just present: await the element you are about to use, never click on a fixed sleep.
2. Read the page first: structure, headings, controls, current URL. Build the selector plan from what is actually there (snapshot refs, roles, labels), not from guesses.
3. Note the auth state and anything that will swallow your first click (cookie banners, interstitials, modals).

## Act, then capture the state

- After every action capture BOTH: the action performed and the resulting state (URL, the target element's text or attributes, visible feedback). "Clicked button" and "order confirmed, ref #1234 shown" are different evidence.
- Verify the intended effect, not just the absence of an error: row created, redirect happened, message shown, count changed.
- Re-read the page after anything that navigates or re-renders; stale element refs are the top cause of flaky interaction.
- Check side effects you can reach: the list the form wrote to, the request that was sent, the message queued.

## Console, network, screenshots

- After the flow, read the console logs. Errors and failed requests are defects until explained; correlate UI symptoms with the console entry that caused them.
- Screenshot the settled state (after animations). When a flow has loading and populated phases, both are evidence; keep each.
- Screenshot at a width that shows the defect, narrow for reflow bugs, and keep the frame next to the claim it supports.

## Rig hygiene

- Reuse ONE browser instance across a run's cases. Never launch a fresh browser per case, and never install or script a browser engine: the harness tool drives the user's real browser with their logins intact.
- Headless stays headless: no windows raised, no focus stolen, no visible tabs spawned per case.
- Scripted rigs: tear down by process group, reaped by exact pid. Close the tab you opened when the run ends.

## Evidence format

Per case: URL, action, expected vs actual (paste the strings), console errors, screenshot path. A finding that names these can be re-run by someone else.

## Red flags

- Claiming a flow works from a happy path only; error and empty cases are the ones that break.
- A selector that matched several elements; you clicked "the first button" and do not know which.
- A screenshot taken before the state settled, or of the wrong viewport.
- Reproducing a bug without capturing the reproduction (nobody can check the fix).
- Leaving tabs, processes, sessions, or locks behind after the run.
