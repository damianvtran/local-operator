# Highlights: reliability and prepared documents

Highlights supplement a completed real-user answer; they never replace it or
modify its model route. Graphics must be enabled. Headless hosts, subagents,
background wakes, failed turns and the feature kill switch retain their existing
exclusion rules. Generated media stays in its existing transcript tiles.

## Automatic graphics

Automatic graphics remain evidence-driven: a numeric table/data result can justify
a chart, ordinary prose alone cannot. `supplements.model: auto` prefers the resolved
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
do not mean exactly two HTTP requests. No errand-owned generic retry/backoff loop
or foreground routing mutation is introduced.

The row records model attempts and bounded scrubbed typed diagnosis. Provider
URLs/private paths/credential shapes are not copied to the shared diagnostic.
Legacy rows with only an exception classname cannot recover a historical reason.

## Prepared standalone HTML

A **successful `write` of `.html` or `.htm` in the current logical turn** may publish
an inline preview directly, without a classifier or generator. The exact in-memory
write content is captured (at most three documents, 48 KiB UTF-8 per document,
also inside the turn's aggregate text budget). Nothing rereads arbitrary existing
files or sends their contents to a vendor. Written/resolved root boundaries,
scratch/generated-tree exclusion, sensitive denylist and operator `denyPrefixes`
all apply before promotion. The original file action remains alongside the preview,
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
numeric claims still require the turn's existing data provenance. CSS layout
numbers are geometry, not plotted claims.

For useful designs: use the injected theme variables, content-sized responsive
layout, native keyboard-operable controls with visible focus, reduced-motion-safe
interactions and readable data labels. Do not require hover to understand a value.

## File discovery

Known document and HTML edits contribute metadata candidates; source-code edits
remain ordinary tool/diff rows. Relative final-answer deliverable references are
recognized, but already-linked files normally remain deduplicated. Prepared HTML
is the narrow exception above. No recursive directory scan or indiscriminate
read-call harvesting is performed.
