---
name: app-files
description: "Write or edit an application's own file format (project, office XML, image-editor file) so it opens as if the app made it: app ids, object types, round-trip."
---

# Application file formats

Read this before you create or change a file an application owns — a video or
audio project, a slide deck or document's XML, an image-editor document, a
game save or app config — by writing its format yourself instead of through the
app's own UI.

Why: whatever reads the file next (the app, the user, a colleague's script, a
grader) expects what the app's own operation produces — its identifiers,
object types, names and structure. A look-alike that renders the same can
still be wrong: a generic backend filter where the app has a named one, one
merged clip where the user asked for several cuts, a text box where a
placeholder was. These are silent defects; a screenshot will not show them.

## Order of preference

1. **Drive the app**: its UI, or its own scripting, batch or CLI mode (a
   headless export, plugin console, macro), which writes the format natively.
2. **The format's own maintained library**, when the app has no script surface.
3. **Hand-written format** only as a last resort, under the rules below.

## Rules when you write the format yourself

- **Use the app's identifiers and defaults.** Read the app's own metadata
  (filter/effect definitions, schema, presets) or make a tiny sample with the
  app's own action and copy its ids, names and attributes. Never invent a
  plausible id; an id the app merely tolerates is not the object it makes.
- **One app operation, one object.** N requested clips, cuts, layers or slides
  are N objects, as the app would make them — not one merged equivalent.
- **Preserve identity and type when editing.** A placeholder stays a
  placeholder, a link stays a link, a style stays a style: change content,
  not kind. Keep the names the app assigns (a layer loaded from a file is
  named after that file) unless asked to rename.
- **Reference the original sources** the user gave, not transcoded or
  re-encoded intermediates you made.
- **The user's negative constraints bind what you create.** "Don't rename,
  move, resize or re-encode" applies to objects you add, not only existing ones.
- **Edit the named file in place**: the deliverable is the file the user
  pointed at, not a renamed sibling copy.

## Prove it loads as intended

- **Round-trip it through the app**: open the result in the app (or its
  headless loader), save or export, and diff against your file — or diff your
  structure against an app-made sample of the same operation. A difference in
  ids, types, names or counts is a defect to fix, not noise.
- **Check mechanical properties with a command**: parse the file and print
  every object's id, type, name and source, rather than judging a render.
