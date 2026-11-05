# ur cli.team_bridge audit tbb3400 (PR #3400)

Scope: the two added keys `notify.budget_exhausted_title` and `notify.budget_exhausted_body` at PR head `5456b8db1d017c421c7632324ab2065e64e8b85e`, read against the en source, `cli.team_bridge.context.json` notes (budget used up so new agent messages wait; nothing lost, screened automatically as the budget refills; people and messages addressed to the agent screened first; no implication of rejection, deletion or notification), the ur style guide, `i18n/glossary.md`, and the namespace's existing renderings. The other 55 keys are byte-unchanged since the PR #3198 passed row and were not re-audited.

## Translator values (as committed at head)

- title: ایجنٹ پیغامات کی جانچ فی الحال رکی ہوئی ہے
- body: آپ کے ایجنٹ نے اس گھنٹے کا جانچ بجٹ پورا کر لیا ہے، اس لیے نئے ایجنٹ پیغامات انتظار میں ہیں۔ بجٹ دوبارہ بھرتے ہی ان کی جانچ خود بخود ہو جائے گی۔ لوگوں کے بھیجے ہوئے پیغامات اور آپ کے ایجنٹ کو مخاطب پیغامات پہلے جانچے جاتے ہیں۔

## Findings and dispositions (all fixed by this auditor in the audit commit)

- **U1 MAJOR** (title+body) — current: `جانچ`. Namespace screening term is اسکریننگ (status.screening); جانچ is the inbox/readiness check (status.last_check, setup.failed). Fix: اسکریننگ in title and body.
- **U2 MAJOR** (title+body) — current: `ایجنٹ پیغامات`. Namespace product name is 'ایجنٹ کے پیغامات' (19 uses); only the new keys used the bare compound. Fix: 'ایجنٹ کے پیغامات کی اسکریننگ…'; 'ایجنٹ کے نئے پیغامات'.
- **U3 MINOR** (body) — current: `کو مخاطب پیغامات / پورا کر لیا`. Clipped participle; 'پورا کر لیا' reads as 'met the budget'. Fix: 'آپ کے ایجنٹ کو مخاطب کر کے بھیجے گئے پیغامات'; 'پورا استعمال کر لیا'.

## Shipped values after remediation

- title: ایجنٹ کے پیغامات کی اسکریننگ فی الحال رکی ہوئی ہے
- body: آپ کے ایجنٹ نے اس گھنٹے کا اسکریننگ بجٹ پورا استعمال کر لیا ہے، اس لیے ایجنٹ کے نئے پیغامات انتظار میں ہیں۔ بجٹ دوبارہ بھرنے پر ان کی اسکریننگ خود بخود ہو جائے گی۔ لوگوں کے بھیجے ہوئے پیغامات اور آپ کے ایجنٹ کو مخاطب کر کے بھیجے گئے پیغامات کی اسکریننگ پہلے ہوتی ہے۔

Accuracy: all three context facts survive; no rejection, deletion or notification implied. Unresolved blocker/major: 0.

## Mechanical

Key parity 57/57; no placeholders, ICU or markup in either key (none in source); NFC; no Cf/bidi control characters; no code tokens. No `maxChars`/`maxCells` hint exists for these keys; both are inbox-notification prose, not width-bound terminal cells. RTL: no Latin tokens, digits or paths are embedded in the Urdu runs, so there is no bidi-isolation hazard.

## Negative control

At head 5456b8db (rows absent) the generator's `_shipped_locales()` returned `['en']`: the stale `cli.team_bridge` hash would have dropped all seven locales. After these rows it returns all eight (see audit.log).

## Limitations

Independent LLM review, not native-speaker certification; no MT or translation service used. Scalar text only: no rendered inbox, terminal, phone or RTL paint geometry was exercised. The translator's model is not named in PR #3400 and is recorded as unrecorded, not inferred.
