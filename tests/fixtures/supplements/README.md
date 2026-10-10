# Turn-supplement contract fixtures (lane C0)

The frozen contract of `docs/design/turn-supplements.md` (§2.4 row, §2.6 document, §2.7 wire),
as data. Every other lane (engine C1, routes C2, TUI, relay web, UI, native) codes against
these files; none of them is hand-written -- `build.py` generates them from the contract's
own types and the real assembler, and `tests/unit/supplements/test_fixtures.py` fails when the
committed bytes drift from `build.py`.

    .venv/bin/python tests/fixtures/supplements/build.py          # rewrite after a contract change
    .venv/bin/python tests/fixtures/supplements/build.py --check  # exit 1 on drift

| dir | what | consumers assert |
|---|---|---|
| `rows/*.json` | full journal lines (`{id,ts,type:"custom",payload:{custom_type:"supplement_v1",details}}`) | parse + re-serialise byte-stable; `queued_stale.json` is THE stale-row fixture (`state=queued`, no live job) |
| `rows/dispositions.json` | what every surface must paint per row, live job vs cold reader; `files_*` = the row's file callouts above the line (§2.4: files do not wait for the generator) | the stale-row rule and display copy of §2.8 |
| `rows/journal_versions.json` | two versions of one anchor in journal order | newest-version-wins |
| `events/*.json` | `supplement_progress` events, one per state | `SupplementProgressEvent` round trip; `running`/`cancelling` are live-only |
| `messages/messages.json` | host->frame theme/ping; accepted and REJECTED frame->host shapes | the nonce echo and the four accepted shapes `ready|resize|error|pong` |
| `components/*.html` | stored component blobs (`<data>` blocks + body), `digests.json` = sha256[:32] | what `AttachmentStore.put_bytes(raw,"text/html")` holds |
| `documents/{populated,empty,error}.html` | the assembled documents, byte-exact | identical bytes on every surface |
| `geometry/long_unit_220.json` | the 220 px long-unit chart of design round 3 (D3-2) | label boxes stay inside the frame; the one known overprint is pinned |

`documents/*` embed the vendored prelude, so they change whenever the prelude does
(`PRELUDE_VERSION` bump + regenerate).

Lane C1a adds `golden/turns.json`: 60 labelled turns (user message, final answer, tool call and
result, a file tree to materialise) used by `tests/unit/supplements/test_golden_set.py` to
measure the spam rate, the pre-filter absorption and the graphics gate (design §5.2). It is
HAND-AUTHORED data, not generated -- `build.py` does not own it, and the drift test globs only
the generated directories.
