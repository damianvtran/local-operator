# Browser QA recipes

Sequences for common cases. Keep the order: wait, read, act, capture. Adapt action names to the harness `browser` tool you have.

## Form submission with validation

1. Open the form; read it and list every field with its label.
2. Submit empty. Capture the validation messages; each required field should be named.
3. Fill one field with an invalid value (bad email, past date). Capture the specific message, not "validation failed".
4. Fill valid; submit; await the success state (do not sleep).
5. Capture: resulting URL, success text, and one reachable side effect (record in the list, message queued, count changed).

## Login-gated page

1. Open the app root; if redirected to login, note both URLs.
2. Log in. If credentials are not yours to hold, ask the operator to sign in by hand; never script around a missing login.
3. Re-request the original deep link: it must land where intended, and a reload must keep the session.
4. Check the logged-out case too: the deep link must redirect, not render a broken shell.

## SPA navigation

1. Note the element that will change (main heading, list count) before clicking.
2. Click the nav item; await the new heading, not a timeout.
3. Capture URL, heading, and count. Then go back; it must restore the previous view. Test both directions.

## Reproducing a front-end bug

1. Reduce to the smallest steps: exact URL, exact sequence, exact data.
2. Capture the console and the failing request at the moment it happens.
3. Write observed vs expected, one line each. Fix, then rerun the same steps and show them passing.

## File upload and download

1. Upload: try one file over the size limit and one wrong type; capture each message.
2. Upload valid: confirm the resulting state (thumbnail, entry, count).
3. Download: the file must arrive with the right name and content type; open it and verify the contents, not just the click.

## Responsive defect

1. Reproduce at the narrowest supported width first.
2. Capture the frame, the overflowing element (scrollWidth vs clientWidth), and the computed sizes involved.
3. Widen stepwise to the width where the break disappears; that number goes in the report.
