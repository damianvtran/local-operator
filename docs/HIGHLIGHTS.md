# Highlights: reliability and prepared documents

Highlights supplement a completed real-user answer; they never replace it or
modify its model route. Graphics are enabled by default (`supplements.graphics`; off with
`false`, or everything off with `supplements.enabled: false` / `LOP_SUPPLEMENTS=0`). A
real-user turn includes one resumed by the operator's *answered* ask. Headless hosts, subagents,
background wakes, failed turns and the feature kill switch retain their existing
exclusion rules. Generated media stays in its existing transcript tiles.

## Automatic graphics

Automatic graphics remain evidence-driven: a numeric table/data result can justify
a chart. A list-structured answer with four explicit operational entity counts
(comments / profiles / links / matches) can also justify a small labelled summary table:
heterogeneous units are marked table-only, never a comparable bar series or a total.
Single numbers, versions and ordinary prose still cannot. `supplements.model: auto` prefers the resolved
operator `hi` tier (including its full provider-qualified model id), then a shipped
registry-known design model, then the effective conversation model. Credential
presence is not current quota availability.

On a typed authentication, quota, transient, timeout or specific model-unavailable
failure, the errand may invoke **one distinct alternate model**. The effective
conversation model is the preferred recovery target when distinct because it is
already serving the user. Explicit `provider/model` and `session` settings are
strict and never substitute another model. All completion invocations share the
same `maxTurns`, deadline and cumulative measured usage/cost. Default two turns
can therefore cover failure + recovery, not also a repair. An isolated transport
may itself perform its existing bounded auth re-resolution; two model invocations
do not mean exactly two HTTP requests. Unknown pricing or missing usage after provider dispatch blocks additional paid invocations; a provider receipt (including explicit $0), known local-free provider, or proven pre-dispatch credential refusal is distinct from unknown cost. No errand-owned generic retry/backoff loop
or foreground routing mutation is introduced.

The row records model attempts and bounded scrubbed typed diagnosis. Provider
URLs/private paths/credential shapes are not copied to the shared diagnostic.
Legacy rows with only an exception classname cannot recover a historical reason.

## Prepared standalone HTML

A **successful `write` of `.html` or `.htm` in the current logical turn** may publish
an inline preview directly, without a classifier or generator. The exact in-memory
executed write content is captured privately before history argument redaction (at most three documents, 48 KiB UTF-8 per document,
also inside the turn's aggregate text budget). Nothing rereads arbitrary existing
files or sends their contents to a vendor. Written/resolved root boundaries,
scratch/generated-tree exclusion, sensitive denylist and operator `denyPrefixes`
all apply before promotion. Later successful edits/uncaptured overwrites invalidate every spelling of the resolved document identity, including symlink aliases; shell/eval conservatively invalidate pending snapshots. Raw bytes never enter model history, diagnostic events or vendor requests. Accessible title/source metadata separately uses the authoritative secret scrubber and descriptive fallbacks; private authored body bytes are not rewritten. The original file action remains alongside the preview,
even when the answer explicitly links the newly prepared HTML.

Supported documents have `<!doctype html>`, an optional `lang` on `<html>`, one
`<head>` and `<body>`, and one plain-text `<title>` used as the accessible frame
name. Head content supports UTF-8 charset metadata and inline style/script. Markup,
inline styles and scripts retain their authored order. The adapter wraps only this
supported profile in the existing component validator and attachment store; the
existing opaque-origin sandbox, no-network CSP and host handshake stay unchanged.

External/sidecar scripts, stylesheets, CSS imports/resource URLs, module loaders,
base URLs, meta refresh/custom CSP and unsupported head forms are refused rather
than silently stripped. Multifile projects and arbitrary websites remain file
callouts. A refusal does not affect opening/revealing the original file. Inline
data images remain permitted by the existing CSP. Qualitative diagrams are valid;
numeric claims use a component-local value set copied from existing turn evidence, without requiring proprietary authored markup. Unsupported fabricated values still fail; step ordinals and timer delay arguments are not data claims. CSS layout
numbers are geometry, not plotted claims.

For useful designs: use the injected theme variables, content-sized responsive
layout, native keyboard-operable controls with visible focus, reduced-motion-safe
interactions and readable data labels. Do not require hover to understand a value.

## File discovery

Known document and HTML edits contribute metadata candidates; source-code edits
remain ordinary tool/diff rows. Relative final-answer deliverable references are
recognized, but a file the turn only *mentions* that the answer already links remains
deduplicated. A deliverable the turn *wrote* (a `write`, or a shell/python output) is kept
even when the answer links it, and the row marks it `linked_in_answer`.

Shell and python writes are read with the command's `cd` tracked (heredoc bodies fed to an
interpreter are scanned for `open(..., 'w')`/`to_csv`/`write_text` literals; `cp`/`mv`/`tee`/
redirect destinations count). A python write through a *variable* path arms a one-level,
<=200-entry look at the directory the command worked in, keeping only visible
deliverable-kind files (and checksum manifests such as `SHA256SUMS`) modified during the
turn. No recursive scan and no read-call harvesting is performed. Every exclusion
(scratch, `.env`/credentials, `*.log`, outside the cwd and `~`, `denyPrefixes`) applies to
these doors exactly as to the old ones.


Desktop cold/older-page and SSE-open snapshots declare `supplements=1` alongside `open_frame=1`. Only a capable reader and that specific viewer receive bounded `supplement_v1` metadata; unknown custom types and errand prompts remain excluded. Component bodies remain authorized digest-addressed attachments, never embedded in paintable history. Older viewers omitting the declaration retain their prior page.

### Table recall versus chart precision

Graphics still need a real user turn, grounded evidence and a vendor answer. Charts/series
retain probability >=0.7. A validated **labelled-summary/table-only** dataset of four exact,
distinct operational entity counts uses `SUMMARY_TABLE_THRESHOLD=0.6`: collecting explicit
labels/values into a table has lower false-positive harm than inferring a ranking/trend across
comparable bars. Mixed datasets do not get this threshold. Conservative uncertainty guards,
no-vendor refusal and all timeout/cost/kill switches remain. This is an explicit recall tradeoff,
not a claim that prior live p=.60/.65 decisions passed the old .7 gate. The prior explicit-table
generator pilot cost computed $0.004602 for1741/112 tokens in4.150s; it was not automatic proof.
A subsequent whole-Session automatic sanitized replay at the new policy returned real vendor
p=.68, then generated and persisted one labelled table in one turn:1764/121 tokens, computed
$0.004738, answer settled0.419s and component/file row settled4.691s from pilot start. All four
files were offered; the vendor selected only the primary receipt. These are observations,
not fixed latency/cost promises; the decision vendor supplied no dollar receipt.
