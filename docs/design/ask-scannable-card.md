# Design: making the `ask` card scannable again

Status: design recommendation, ready to hand to a coder. Author: designer.
Base: worktree `lo-ask-scroll`, branch `fix/ask-long-text`, HEAD `97881577`.
Supersedes the description-layout half of `docs/design/ask-long-descriptions.md`
(§1 (A)); it does **not** supersede that document's reveal design (§1 (B)),
which survives with one changed job.

Every frame named here is committed under `docs/design/frames/ask-scannable/`
and was rendered from the scripts in `scripts/`, then viewed as PNG via
`qlmanage -t -s 1600`. Every number was measured from the exported frames, not
estimated from the code. The mock frames were produced in a throwaway
`git worktree` at `/tmp/lo-mock` and the worktree was removed; the shared
checkout was never edited.

---

## 0. Verdict up front

The user is right, and the defect is larger than "it looks busy".

**Three separate defects, two of them functional, not cosmetic:**

| id | severity | what |
| --- | --- | --- |
| D1 | BLOCKER | At 190x50 and 150x40 the card is a wall of prose. Labels do not win the visual hierarchy and the list cannot be counted at a glance. |
| D2 | BLOCKER | `ctrl+e` is **dead** at 190x50 and 150x40 — the frame is byte-identical before and after the press, and the footer does not offer `^e`. At 150x40 two descriptions are still ellipsised, so the text is unreachable at the size the user works at. This is the ORIGINAL bug, unfixed. |
| D3 | MAJOR | At 150x40 one arrow press rewrites **9 of 20 card rows**. The prose pool redistributes between rows as the cursor moves. The card's height is stable, so the existing tests pass, but the body churns. |
| D4 | MAJOR | `recommended` renders at `muted` — the identical style to the prose it sits in — with no weight. It does not read as a badge. |

D2 is the finding I would escalate first. The round shipped a reveal key that
does not fire on the two largest terminal sizes tested, and the reported
screenshot is one of them.

---

## 1. Judging the current frame

### 1.1 190x50, paragraph descriptions — `current-190x50.svg`

The card occupies rows 19-42: **24 body rows**, of which 3 are chrome and
**~19 are prose**. Above it sit **13 completely empty transcript rows.**

```
19| the agent needs your decision
21| Which rollout strategy should we use for the analytics recorder migration?
23| ❯ 1. Migrate in place on open
24|   recommended · The store upgrades itself the first time a session opens it: `_migrate` runs an idempotent…
25|   `_MIGRATION_COLUMNS` that is not already present, each carrying a DEFAULT so rows written by older…
26|   columns already took, so it is well-trodden here, and it means a user who upgrades mid-week never has…
27|   widens the table and everything downstream keeps working. The cost is that the migration runs on the…
28|   change on a very large ledger could stall the recorder's queue for a noticeable interval; in practice…
29|   even on a multi-megabyte database, which is why the existing code takes this route and why it remains…
30| 2. Rebuild the ledger into a fresh file          <-- a LABEL, visually identical to the prose above it
31|   Create a new database alongside the old one, copy rows across with the new columns computed rather…
...
42| ↑↓ move · 1-9 jump · enter answer · esc skip     <-- no `^e`
```

What I saw in the rendered PNG: option 1 gets **7 lines**, option 2 gets **4**,
option 3 gets **3**. The labels are `fg` bold and the prose is `muted`, so the
contrast ranking is technically intact — but a bold line every seven lines,
flush against the paragraph above it with **no blank line between rows**, does
not separate anything at this density. My eye ran straight from the end of
option 1's paragraph into option 2's label and kept reading it as the same
block of text. The row-3 label ("Version the columns and read both shapes") is
the worst case: it sits immediately after a full-measure line of option 2's
prose that ends mid-sentence.

At 186 cells wide the prose is also **far past a readable measure**. Typography
convention is 45-75 characters per line; these lines are ~180. That alone would
make the block hard to scan even without the separation problem.

**Counting the options at a glance: no.** Three options plus free-text, spread
over 19 rows with no visual rhythm. The ordinals `1.` `2.` `3.` are drawn at
`dim` (3.43:1) — the weakest thing on the row, so the one affordance that could
carry the count is the one the eye finds last.

**Reading the one I care about: yes, but only because it is first.** Option 1
is complete. Options 2 and 3 are complete here too — at 190x50 the pool covers
everything.

**D2 confirmed here.** `scripts/ask_user_repro.py 190x50 0 reveal` produces a
frame with **md5 `e1823d51…`, identical to the non-revealed frame**. The footer
never offers `^e`. `_reveal_is_useful` (ask_picker.py:2753) requires the
selected row's description to be CUT; at 190x50 the pool has already drawn it
in full, so the key is correctly withheld. The reveal is working as designed —
the design is wrong, because the wall is exactly the state where the user most
wants to collapse it, and the key that would is refused.

### 1.2 150x40 — `current-150x40.svg`, `current-150x40-cursor2.svg`

This is the worst frame of the set, and it is the one where the shipped
behaviour is actively incoherent.

Card is rows 13-32, **20 rows**. Cursor on row 1 (the default, `recommended=0`):

- option 1 gets 3 lines, **ellipsised** (`…releases r…`)
- option 2 gets 2 lines, **ellipsised**
- option 3 gets 1 line, **ellipsised**
- footer: `↑↓ move · 1-9 jump · enter answer · esc skip` — **no `^e`**

So at 150x40 the card **truncates three of three descriptions and offers no way
to read them.** That is the bug the round set out to fix, still present, now
with more lines spent on it. The reveal is withheld because
`_layout(reveal=True).reveal_rows >= 1` fails — the pool has already eaten the
budget the reveal block would need. The two halves of the shipped design fight
each other: (A) spends the leftover budget, which starves (B).

**D3 measured here.** Moving the cursor from row 1 to row 2:

| size | rows changing content on one arrow press |
| --- | --- |
| 190x50 | 2 / 24 (cursor only — fine) |
| **150x40** | **9 / 20** |
| 130x30 | 2 / 10 (cursor only — fine) |

At 150x40 the selected row's grant grows from 3 lines to 6 and every other
row's shrinks to pay for it, so nearly half the card rewrites itself under the
user's eye. Card HEIGHT is constant (17-20 rows, verified stable across all
four cursor positions at every size), which is why the design's own
anti-reflow test passes — but "the card does not change height" was a proxy for
"the card does not move", and the proxy has come apart. The user sees text
churn on every keypress.

### 1.3 130x30 — `current-130x30.svg`, `current-130x30-reveal.svg`

**This frame is good, and it is the proof that the fix is a cap and not a
rewrite.** Budget is too tight to buy any continuation lines, so the card falls
back to label-only:

```
17| ❯ 1. Migrate in place on open  · recommended
18|   2. Rebuild the ledger into a fresh file
19|   3. Version the columns and read both shapes
20|   4. Other (type your own)
22| ↑↓ move · 1-9 jump · enter answer · ^e more · esc skip
```

Four options, four rows, countable instantly, and `^e more` is offered and
**works** (reveal frame differs, shows 5 lines of prose for row 1). This is the
scannable card. The list has shape. The reveal earns its keep. Note this is
also the only size where the tag is drawn on the LABEL line (` · recommended`)
rather than in the prose.

The irony is precise: **the card is best where the budget is smallest.** The
shipped design spends every spare row it has making the frame worse.

### 1.4 Short two-clause descriptions — `current-short-190x50.svg`

`scripts/ask_shot.py` at 190x50. Five options, each one label line plus one
short prose line, and it is **completely fine**:

```
1. Drop the rows
   nothing reads the column any more
2. Backfill from the audit log
   recommended · slower, keeps history
3. Dual-write for a week
   safest, needs a follow-up MR
```

The label/prose alternation is a legible 2-line rhythm. I can count five
options without reading a word.

**So: is there a description length below which the current behaviour is
right?** Yes, and it is sharp. The behaviour is right at **1-2 wrapped lines per
row** and degrades from 3. At 2 lines the label/prose pair still reads as one
unit; at 3+ the paragraph outgrows its label and the alternation stops being
perceptible. That threshold is the recommendation in §2.

### 1.5 The approval gate — `current-approval-100x30.svg`

Fine today, exactly as the prior design said. Three options, consequences of
37/36/28 cells, one line each, perfect 2-line rhythm, no ellipsis, no tag
(it passes no `recommended`). **This frame is the constraint, and I verified
every proposal against it byte-for-byte.**

---

## 2. Recommendation

**Cap the default view at 2 description lines per row, and let `ctrl+e` be
what uncovers the rest.**

One change to the allocator; the reveal design already in the tree is kept and
finally becomes reachable.

### 2.1 What to change

In `_layout` (ask_picker.py:1700-1731), the continuation-line pool loop:

```python
extra_lines = len(self._description_lines(index, width)) - 1
```

becomes

```python
extra_lines = min(len(self._description_lines(index, width)), DEFAULT_DESC_CAP) - 1
```

with `DEFAULT_DESC_CAP = 2`, and the matching cap in the `described` term of
the natural-height calculation (ask_picker.py:1439-1442) so the card does not
request rows it will never spend:

```python
described = sum(
    max(1, min(DEFAULT_DESC_CAP, len(self._description_lines(index, self._card_width()))))
    for index in range(self.row_count)
)
```

`DESC_MAX_ROWS = 6` stays as the REVEAL cap. The two caps are different numbers
for different jobs: 2 is "how much prose the list shows per row", 6 is "how much
prose the reveal shows for one row".

### 2.2 What it does — measured, not predicted

I implemented exactly this in a throwaway worktree and rendered it.

`proposed-cap2-190x50.svg`, cursor on row 1:

```
26| the agent needs your decision
28| Which rollout strategy should we use for the analytics recorder migration?
30| ❯ 1. Migrate in place on open
31|   recommended · The store upgrades itself the first time a session opens it: `_migrate` runs an idempotent…
32|   `_MIGRATION_COLUMNS` that is not already present … This is the path the cost co…
33|   2. Rebuild the ledger into a fresh file
34|   Create a new database alongside the old one, copy rows across with the new columns computed rather…
35|   verified. This is the only option that can backfill a column … has to be r…
36|   3. Version the columns and read both shapes
37|   Leave every existing database exactly as it is and teach the read path to tolerate either shape…
38|   them, and fall back to a computed expression otherwise … a user who rolls back…
39|   4. Other (type your own)
40|   an answer that is not on the list — type it here
42| ↑↓ move · 1-9 jump · enter answer · ^e more · esc skip
```

| | current | proposed | change |
| --- | --- | --- | --- |
| card height, 190x50 | 24 rows | **17 rows** | −7 |
| transcript rows visible, 190x50 | 13 (all blank) | **20** | +7 |
| rows per option | 8 / 5 / 4 | **3 / 3 / 3** | uniform |
| `^e` offered at 190x50 | **no** | **yes** | D2 fixed |
| `^e` offered at 150x40 | **no** | **yes** | D2 fixed |
| rows churning per arrow press, 150x40 | **9 / 20** | **2 / 17** | D3 fixed |
| card height stable across cursor moves | yes | yes | preserved |
| approval gate at 100x30 | — | **byte-identical** | no regression |
| short-description card at 190x50 | — | **byte-identical** | no regression |

The last two rows are the ones that matter for safety, and I checked them by
diffing the extracted text of every row of the frame, not by inspection: the
approval gate and `ask_shot.py`'s two-clause card are **unchanged**, because
neither ever asks for more than 2 description lines. The cap is invisible to
every frame that was already correct.

Judged against the three questions:

- **Can I count the options at a glance?** Yes. Uniform 3-row blocks
  (label + 2 prose). The rhythm is perceptible in peripheral vision.
- **Can I read the one I care about?** The first 2 lines inline, the rest on
  `^e`, which is now actually offered. Better than today at 150x40, where the
  answer is currently *no by any means*.
- **Does the card stay still as I move?** Yes — better than today. The grant is
  no longer cursor-dependent at any size that can afford 2 lines for every row,
  so the pool stops redistributing.

### 2.3 Costing the other candidates

| direction | verdict | why, from the frames |
| --- | --- | --- |
| **per-row cap at 2, rest behind `ctrl+e`** | **ADOPT** | measured above. Fixes D1, D2, D3 together; regresses nothing. Smallest diff of the four. |
| blank-line separation between rows | **adopt as a follow-up, not now** | It is the right instinct for D1 and it is what makes a 3-line cap viable later. But at 190x50 it costs 3 rows (one per option) on a card already 7 rows too tall, and at 150x40 it would push the budget back into the starve-the-reveal state that causes D2. Revisit once the cap has bought the headroom. See §5. |
| cap at 3 rather than 2 | **rejected for now** | At 150x40 a 3-cap gives 4 rows/option × 4 options = 16 rows against a 20-row budget, leaving 1 row for the reveal block — right back at the D2 boundary. 2 is the value that keeps `^e` affordable at the size where it is most needed. |
| show prose only for the selected row | **rejected** | The prior design rejected it for height reflow; that objection stands and I re-measured it. But there is a second objection it did not name and I consider decisive: it removes the non-selected consequences from the APPROVAL gate, where all three are the authorisation text. `ApprovalPrompt` subclasses this card, so a collapsed-list default reaches it. The prior design's §0.2 is right about this. |
| dim non-selected prose | **rejected** | Prose is already at `muted` = 6.51:1. The next step down is `dim` = **3.43:1**, under the 4.5:1 AA floor the description was deliberately walked UP to (D7, round 1). On the approval gate that would put two of three authorisation consequences below AA. Contrast is not available to spend here — the ramp has no room left underneath. Making the LABEL louder is not available either: labels are already `fg` at 11.30:1, the top of the ramp. |
| do nothing | **rejected** | D2 alone: the reported size still truncates text with no reveal. |

Note the through-line: **the hierarchy problem cannot be solved with colour on
this card, because both ends of the ramp are already spent.** It has to be
solved with SPACE — fewer prose lines per row — which is what the cap does.

---

## 3. The `recommended` tag

### 3.1 Confirming the complaint

Confirmed at ask_picker.py:2454-2455. The tag is appended with
`style=ground + tag_ink`, and the call site (ask_picker.py:2123) passes
`muted` for both `tag_ink` and `ink`:

```python
for line in self._description_text(index, width, ground, muted, muted, granted):
```

So the badge is **the same colour as the prose after it, with no weight**. The
docstring at ask_picker.py:2433 claims "At `muted` it is the loudest thing on a
line of `dim` prose" — that was true when prose was `dim`, and stopped being
true when prose was walked up to `muted` in round 1. The comment is now stale
and describes a hierarchy the frame does not have. In `current-190x50.svg` the
word "recommended" is genuinely invisible in the wall; I could not find it
without searching.

### 3.2 Hue is not available — measured

I computed contrast for every candidate hue against BOTH themes and BOTH
grounds (`overlay` for a normal row, `raised` for the selected row):

| treatment | dark/overlay | dark/raised | light/overlay | light/raised | min | AA 4.5:1 |
| --- | --- | --- | --- | --- | --- | --- |
| `muted` (today) | 6.51 | 7.24 | 5.18 | 5.88 | 5.18 | pass |
| **`fg` (label strength)** | **11.30** | **12.56** | **10.92** | **12.39** | **10.92** | **pass** |
| amber / warning | 7.09 | 7.89 | 3.97 | 4.50 | 3.97 | **FAIL** |
| label violet | 5.19 | 5.77 | 3.62 | 4.11 | 3.62 | **FAIL** |
| signal blue | 5.59 | 6.21 | 3.54 | 4.02 | 3.54 | **FAIL** |

**Every hue candidate fails AA on the light theme.** The warm-paper ramp's
chromatic tokens are tuned against `surface`/`paper`, not against the card's
`overlay`, and they lose 3 points of ratio there. And `$lo-accent` is
unavailable regardless: local_operator.tcss:28-46 spends it on exactly four
sites with an exhaustive grep-able list, and site 4 is already ON this card
("what ENTER WILL TAKE in a picker"). A recommendation is not what Enter takes
— it is what Enter takes *by default*, which is a different sentence, and
painting both green would make the accent say two things on one frame.

### 3.3 Proposed treatment

**`▸ RECOMMENDED`, drawn at `fg` + bold**, keeping its current position at the
head of the description's first line.

Three signals that are not hue: **weight** (bold, nothing else on the prose line
is), **case** (uppercase, unique on the card), and a **glyph** (`▸`, which reads
as a marker not a word). Exact tokens for the coder:

```python
RECOMMENDED_TAG = "▸ RECOMMENDED"

# in _description_text, position == 0 and self.question.recommended == index:
body.append(RECOMMENDED_TAG, style=ground + fg + Style(bold=True))
#                                            ^^ `fg`, NOT `tag_ink`/`muted`
```

`_description_text` currently receives `muted` for `tag_ink`; the call site at
ask_picker.py:2123 should pass `fg` for the tag and keep `muted` for the ` · `
separator and the prose. That keeps the separator and prose exactly as they are.

Rendered as `proposed-badge-short-190x50.svg` and `proposed-badge-190x50.svg`.
In the short-description frame the badge is unmistakable at a glance and still
sits below the label in the reading order, because the label owns the line
above it and the badge is indented under it.

**Constraints checked:**

- **Accent rule**: not touched. No new `$lo-accent` site; a grep for `accent`
  in `ask_picker.py` is unchanged.
- **Contrast floors**: `fg` is 10.92:1 at worst across both themes — the top of
  the ramp, above the label's own documented 11.30:1 figure on dark. Prose
  stays at `muted` 6.51:1. Nothing moves down.
- **Hierarchy**: label (`fg` bold, line N) > badge (`fg` bold, line N+1,
  indented) > prose (`muted`). The badge matches the label's weight rather than
  exceeding it, and loses on position — which is the correct ranking for a hint
  that qualifies a label.
- **Approval gate**: `approval.py:965-976` builds its `AskQuestion` with **no
  `recommended` argument**, so `self.question.recommended` is `None` and
  `recommended == index` is never true. I verified this empirically: the
  approval frame at 100x30 is **byte-identical** with the badge change applied.
  Nothing about this treatment assumes a recommended index exists.
- **Width**: `▸ RECOMMENDED` is 13 cells against 11 for `recommended`, +2. The
  reservation at ask_picker.py:1265 reads `cell_len(RECOMMENDED_TAG)`
  dynamically, so it stays consistent with no second edit. If the 2 cells are
  judged too expensive on the narrowest cards, plain `RECOMMENDED` at 11 cells
  is exactly cost-neutral and keeps two of the three signals.

---

## 4. Ordering: should `recommended` lead the list?

**No. Neither the card nor the tool schema should enforce or encourage
recommended-first, and nothing should ever reorder options.**

### 4.1 What comparable systems do

The prevailing convention in interactive pickers is that a default is a
**pointer into the author's order, not a reordering of it**. Inquirer.js — the
most widely used prompt library — treats `default` as an index or value that
sets the initial cursor position only; the choices array renders in the order
given. Its issue #32 exists precisely because users wanted to preselect a
non-first item, and the resolution was to preselect in place. The same shape
holds for this card: `recommended` is `int | None`, documented at
harness/types.py:570 as "0-based index of the option you recommend; it is
preselected and marked", and ask_picker.py:500 preselects it as the cursor.

That is already the right model. The card marks and preselects; it does not
promote.

The convention holds for a substantive reason, not just precedent. Option
order usually carries meaning the author chose — escalating cost, escalating
risk, chronology, or in this card's own approval case a deliberate
allow/deny/allow-all ladder. Promoting one entry destroys that ordering to
gain an emphasis the badge already provides.

Survey-methodology research on **primacy effects** points the same way: options
presented earlier are selected more often, independent of content. That is an
argument for *not* moving the recommended option to the top — doing so stacks a
positional thumb on the scale on top of an explicit recommendation and a
preselected cursor. The recommendation is already carried three times (badge,
cursor, Enter default). A fourth, invisible, positional nudge is not
transparency, it is pressure.

### 4.2 Is reordering behind the model's back a real hazard?

**Yes. Unambiguously, and it is the strongest argument in this section.**

The card draws ordinals `1.` `2.` `3.` and binds digit keys to them
(`1-9 jump` in the footer). Those ordinals are positional. If the card
reordered `options` for display, then:

- the user reads "option 2" and presses `2`;
- the answer is reported back against the option that was at display position
  2, which is a **different element of `options`** than the model's index 2;
- the model wrote the question with a specific `options` order and reasons about
  its own indices — `recommended=0` refers to `options[0]`, not to whatever the
  card floated to the top.

The result is a silent semantic mismatch on the one surface whose entire job is
to transmit the user's decision faithfully, on a card whose sibling is an
**authorisation gate**. There is no repair for it at the display layer: any
reorder needs an index map, and the moment an index map exists, every
transcript, every log line, every `recommended` value and every human reading
the JSON is looking at one of two orderings with nothing marking which.

So: do not reorder, and do not add a schema rule encouraging authors to put the
recommended option first either. A `description` in the schema saying "list the
recommended option first" would push models to bury a natural ordering
(cheapest→safest, least→most destructive) for no gain the badge does not already
deliver. Leave harness/types.py:570 as it is.

**One thing worth telling the user plainly:** the current behaviour already
answers their question correctly. `recommended` sits in its natural position,
is marked, and is preselected. The reason it did not *feel* that way is D4 —
the badge was invisible — not the ordering. Fix the badge and the ordering
question dissolves.

---

## 5. Verdict and follow-ups

**This round is NOT terminal.** Two BLOCKERs (D1, D2) and two MAJORs (D3, D4)
are open, and all four are fixed by the two changes in §2.1 and §3.3.

Hand to a coder as **two independent slices** — they touch different methods
and can run in parallel:

1. **Allocator cap** — `DEFAULT_DESC_CAP = 2`, applied in the pool loop
   (ask_picker.py:1727-1731) and the natural-height `described` term
   (ask_picker.py:1439-1442). Fixes D1, D2, D3. Regression evidence required:
   `approval_shot.py 100x30` and `ask_shot.py 190x50` must stay byte-identical.
2. **Badge treatment** — `RECOMMENDED_TAG` and the `tag_ink` argument at the
   ask_picker.py:2123 call site. Fixes D4. Regression evidence required:
   approval frame byte-identical (no `recommended` index).

Follow-ups, not blocking:

- **F1 (MINOR)** — blank-line separation between option blocks, once the cap has
  bought the headroom. Best long-term answer to D1; costs one row per option.
- **F2 (MINOR)** — line measure. At 190 columns prose runs ~180 cells per line,
  far past the 45-75 readable measure. Consider capping the description column
  (not the card) at ~100 cells. Would also make F1 cheaper by shortening wraps.
- **F3 (MINOR)** — the reveal block currently repeats the 2 lines already shown
  inline, so `^e` at 190x50 shows lines 1-2 twice. Visible in my mock's reveal
  frame. Either start the block after the granted lines, or accept the repeat as
  "here is the whole thing" — but decide it deliberately.
- **F4 (MINOR)** — ordinals are `dim` (3.43:1), the weakest element on a row
  that the footer advertises digit keys for. Below AA for text that names an
  affordance.
- **F5 (NIT)** — the stale docstring at ask_picker.py:2433 ("the loudest thing
  on a line of `dim` prose") describes a hierarchy that stopped existing in
  round 1. Correct it with whichever slice touches that method.
- **F6 (NIT)** — `scripts/ask_user_repro.py` is **untracked** in the worktree
  (`git status` shows `??`). It was described to me as committed. Commit it, or
  the frames in this document cannot be reproduced by anyone else.
