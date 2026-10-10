# Copy checks

Deterministic scans over user-visible strings. Run before judging tone; each defect gets a location and the offending string.

## Scan for

- **Placeholders**: "TODO", "lorem", "XXX", "FIXME", "undefined", "null", raw template braces.
- **Generic errors**: "Something went wrong", "An error occurred" with no cause and no next step.
- **Dead ends**: an empty or error state that does not say what to do next; link text whose destination is missing or dead.
- **Ambiguity**: one label that could mean two actions ("Submit" where "Save draft" or "Publish" is meant); unlabelled icons with no tooltip.
- **Inconsistency**: the same concept named two ways (Save, Store, Apply); casing and punctuation drifting within one screen or flow.
- **Truncation risk**: strings that clip at the narrow viewport; test the longest real value, not the demo data.
- **Pluralisation and numbers**: "1 items", "0 item(s)"; raw numbers without units; dates without a format a global audience can read.

## How to run

- Web: read the rendered page and search the visible strings for the patterns above. Check the `lang` attribute and the page title.
- Terminal: copy the actual pane text and search it; include help and status strings.
- Code: grep the changed files for the pattern list where the change touches user-visible strings.

## Judgement (after the scan)

- Does the copy answer the user's question at that moment: what happened, why, what next?
- The audience's vocabulary, not the implementation's: no internal names, no stack traces in user copy.
- Claims must be true of the product as shipped. A "saves automatically" caption next to a manual save button is a defect.

## Evidence format

Per item: location, the string, and the proposed wording. Example: `settings.html:42 "Something went wrong" -> "Could not save: check your connection and retry"`.
