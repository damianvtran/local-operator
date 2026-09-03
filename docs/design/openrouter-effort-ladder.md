# Listing-derived reasoning-effort ladders — design

Status: proposal for one coder PR. Author: architect (lopdev team), 2026-09-03.
Cites the working tree at `origin/main` (`ecabd0de`) and a live
`GET https://openrouter.ai/api/v1/models` pull taken 2026-09-03 (424 rows).

> **AMENDED after round-1 QA (2026-09-03). One decision in this note was
> REVERSED by measurement: the listing's `default_effort` is NOT seeded.**
>
> §5.1 recommended seeding it, resting on the inference that sending
> OpenRouter's own stated default is equivalent to omitting the key. QA put
> that on the wire and **disproved it**: `z-ai/glm-5.3` seeded at `max`
> returned 2.16× the reasoning tokens of omission (medians 200 vs 92.5, n=12
> per arm, non-overlapping distributions), and the effect is model-specific
> (`gemini-3.8-flash` measured 0.93×, a genuine no-op), so no general
> equivalence can be relied on. The documented fallback in §5.1 — *"adopt the
> listing's ladder but seed `reasoning_effort=None`"* — is what shipped.
>
> §5.1 below is rewritten to record that result. **The rest of this note stands
> as written**: precedence, route scoping, the plumbing, the ordering rules and
> the `reasoning` flag were all confirmed by review and QA. Anything elsewhere
> in this document that assumes a seeded listing default is superseded by §5.1.

---

## 0. Corrections to the brief

Four claims in the brief are wrong or incomplete; two of them change the design.

**(a) There is a second, independent bug, and it is the more embarrassing one.**
The brief says the table has "arms only for Anthropic `claude-*` generations and
OpenAI". It does — but the Anthropic arms **do not match the ids OpenRouter
publishes**. `_EFFORT_TABLE` (`effort.py:112-136`) spells generations with
hyphens (`claude-[a-z]+-4-6`), because it was written against Anthropic's own
dated snapshot ids. OpenRouter spells the same models with dots. Measured:

```
anthropic/claude-opus-4-6  -> ('low','medium','high','max')     # direct spelling
anthropic/claude-opus-4.6  -> ()                                 # OpenRouter spelling
anthropic/claude-opus-4-8  -> ('low','medium','high','xhigh','max')
anthropic/claude-opus-4.8  -> ()
```

So **8 Anthropic rows** (4.6/4.7/4.8 and their `:batch` twins) silently lose
their ladder on the OpenRouter route today. `effort.py:10-13` claims the table
is keyed on model not provider precisely so "`anthropic/claude-opus-5` through
OpenRouter is the same model with the same knob" — that invariant is **already
broken**, by punctuation, and nobody noticed because nothing tests the dotted
spelling. This matters for Q2: the route invariant is not something the listing
would break, it is something the listing would *repair*.

**(b) "34 have a ladder DIFFERENT from what OpenRouter reports" — correct, and
all 34 are OpenAI.** Every one of the 16 Anthropic rows the table *does* match
agrees with the listing exactly, ladder and default. The disagreement is
entirely the OpenAI arm, which applies one transcribed `gpt-5.4` ladder
(`_OPENAI_GPT5`, `effort.py:105`) to every `gpt-[5-9]` id. The listing says that
is wrong 34 times in both directions: it **widens** 12 (the `gpt-5.6-*` family
takes `max`) and **narrows** 16 (`gpt-5.4-pro` takes only `medium/high/xhigh`;
`gpt-5-pro` takes only `high`). This is the single strongest evidence for
precedence, and it is one-sided: the hand-transcription is wrong only where it
*extrapolated a family*, never where it transcribed a specific model page.

**(c) "3 rows list the `reasoning_effort` PARAMETER but carry no ladder" —
correct, but the important set is a different one.** Ten rows carry a
`reasoning` object with **no** `supported_efforts` while our table *does* give
them a ladder: `openai/o1`, `o3`, `o3-pro`, `o3-mini(:batch)`, `o4-mini(:batch)`,
`o1-pro`, `gpt-5-image`, `gpt-5-image-mini`. These rows also **do not advertise
`reasoning_effort` in `supported_parameters`** — they advertise only
`reasoning`/`include_reasoning`. That is a denial-shaped signal against the
o-series arm, and Q1's three-state question turns on it (§1).

**(d) "Every effort word in the listing is already in `EFFORT_ORDER`" —
correct today, and the failure mode if that changes is worse than the brief
implies.** It is not merely a sorting question. A ladder containing a word
outside `EFFORT_ORDER` makes `resolve_effort` **raise**, not misbehave:

```
resolve_effort('probe','medium') with levels ('low','turbo','high')
-> ValueError: tuple.index(x): x not in tuple      # effort.py:209
```

`effort.py:205-208` guards the *requested* word but not the *ladder's* words.
That path runs on a failover hop (`failover.py:1502`), i.e. on the request that
was supposed to rescue a turn. Filtering at ingest is therefore load-bearing,
not tidiness (§4).

Verified-correct in the brief: 153 rows carry `supported_efforts`; 99 of them
get nothing from our table; all 153 arrive **descending**; no `default_effort`
falls outside its own ladder; every one of the 153 carries a `default_effort`;
only the OpenRouter/Radient listing shape has this field — the cached
anthropic/zai/xai/kimi/alibaba/codex documents and **models.dev** (7,495 models,
`reasoning` is a plain bool, zero models carry any effort key) do not.

---

## 1. Q1 — Precedence: the listing wins where it speaks, and it is three-state

**Decision. The ladder becomes three-state, exactly like `supports_images`. A
listing that states a ladder overrules `_EFFORT_TABLE`. A listing that is silent
defers to it. A listing that states a `reasoning` object *without* a ladder is
NOT a denial — it defers too.**

The evidence for listing-over-table is (b) above: where the two disagree, the
disagreement is 100% concentrated in the one arm that extrapolates
(`gpt-(?:[5-9]|\d{2,})` → one hardcoded ladder), and 0% in the arms that
transcribe a specific model page. The table's own docstring
(`effort.py:15-18`) justifies itself as "transcribed from their docs rather than
guessed" — but `_OPENAI_GPT5` *is* a guess about every future `gpt-5.x`, and
the listing shows it is a wrong one 34 times. Meanwhile the aggregator's claim
is not second-hand about the route it describes: it is the router's own
statement about the request it will accept, and it is what returns 400.

This is the same shape as `limits_from_listing` (`registry.py:242-249`,
`configure.py:679-703`): "our transcription is second-hand, go ask." It differs
in one way worth naming — `limits_from_listing` is a per-row opt-in flag,
because only ten hand-transcribed Claude rows needed it. Here the *entire*
table is second-hand by construction, so a per-arm flag would be a flag that is
always true. Hence precedence is unconditional rather than flagged.

**Why the no-ladder `reasoning` object is not a denial.** This is the one place
I depart from the `supports_images` analogy, deliberately. `_stated_bool`
(`discovery.py:393-403`) can treat `false` as a denial because `false` is an
explicit answer to a yes/no question. `"reasoning": {"mandatory": false}` is not
an answer to "which efforts do you take" — it is an object that answers a
*different* question and omits ours. Reading omission as denial would strip the
o-series ladder (10 rows) on the strength of a key that was never about efforts.
The absent `reasoning_effort` in `supported_parameters` is suggestive, but it is
one aggregator's parameter-passthrough manifest, and our table's o-series arm is
transcribed from OpenAI's own reasoning guide. **Silence defers. Only a stated
ladder overrules.** If we later want to honour that denial, the evidence that
would settle it is a live 400 from `reasoning_effort: "low"` to `openai/o3`
through OpenRouter — which nobody in this repo has run, and which the existing
comment at `clients.py:1176-1183` is careful to scope ("that is the extent of
what was measured").

**Consequence to accept:** on the OpenRouter route, `openai/gpt-5.4-pro` stops
offering `none`/`low`. That is a user-visible *narrowing* and it is correct —
the listing says those rungs 400 there.

---

## 2. Q2 — Route scoping: the invariant is already broken, and per-route is right

**Decision. The ladder is resolved per (provider, model), not per model. The
`_EFFORT_TABLE` fallback stays keyed on the model id, so a model with no listing
keeps exactly the ladder it has today.**

The docstring's invariant (`effort.py:10-13`) is a good default and a bad
absolute. Three findings:

1. It is **already violated** today by id spelling — §0(a). Any fix that keeps
   ladders purely model-keyed must *also* fix the dot/hyphen gap, or 8 Anthropic
   rows stay broken. Listing-derived data fixes it for free.
2. An aggregator genuinely can differ. `openai/gpt-5-pro` on OpenRouter offers
   `high` alone; OpenAI's own model page (per `_OPENAI_GPT5`) offers five rungs.
   Whether OpenAI direct really accepts five is not knowable from here — but
   whether *OpenRouter* accepts five is knowable, it says no, and that is the
   route the request takes.
3. OpenRouter is internally consistent below the model level, so we do not need
   to go finer: checked all 424 rows — `:batch`, `:free`, `:nitro` and `:exacto`
   variants **never** disagree with their base ladder (0 disagreements), and
   `/models/<id>/endpoints` carries no per-endpoint `reasoning` object at all
   (verified on `anthropic/claude-opus-5`: every endpoint reports
   `reasoning: None`). Model-level is the finest granularity the data supports.

**The no-listing case is the one that must not regress**, and it is the common
case: 91 shipped registry rows never fetch a listing at all
(`_listing_can_correct` is false for them — measured), and every direct provider
except Anthropic has no reasoning field in its listing. For all of these the
resolution is unchanged: `supported_efforts(model_name)` off the table. The new
data is strictly *additive* except where a listing states a ladder.

---

## 3. Q3 — Plumbing: the exact fields and merge rules

### 3.1 `DiscoveredModel` (`discovery.py:198-251`)

```python
    #: The effort ladder the LISTING stated, ASCENDING, normalised to
    #: EFFORT_ORDER at ingest. Three-state, for the reason supports_images is:
    #: ``None`` is "the listing said nothing about efforts" and defers to
    #: ``model.effort``'s table, while an empty tuple would be a denial no
    #: listing in this tree actually issues (a ``reasoning`` object without
    #: ``supported_efforts`` answers a different question — see the design note).
    #: The parser therefore never produces ``()``; it produces ``None`` or a
    #: populated tuple.
    reasoning_efforts: tuple[str, ...] | None = None
    #: The rung the listing says the model runs at when nothing is sent.
    #: ``None`` is unstated. Only meaningful beside a populated ladder, and the
    #: parser drops it when it is not a member of that ladder.
    reasoning_default_effort: str | None = None
```

This is the third exception to "0 means unknown" and it is the same exception
`supports_images` and `free` already are: the source said something the struct
could not otherwise record. The docstring's exception list needs a fourth
paragraph saying so.

Note the deliberate asymmetry with `ModelSpec.reasoning_efforts`, which is a
plain `tuple` where `()` means "no knob" (`types.py:1568`). That is right for
the *spec* — the spec is the resolved answer and has no silence to record — and
wrong for the *row*. Naming them the same is intentional; a reviewer should
confirm no code path assigns one to the other without going through the merge.

### 3.2 Parser — `_row_from_openai_entry` (`discovery.py:545`)

```python
    reasoning = _mapping(entry.get("reasoning"))
    ladder = _effort_ladder(reasoning.get("supported_efforts"))
    ...
        reasoning_efforts=ladder,
        reasoning_default_effort=_effort_default(reasoning.get("default_effort"), ladder),
```

with two new coercers beside `_stated_bool`:

```python
def _effort_ladder(value: object) -> tuple[str, ...] | None:
    """A listing's effort list as an ASCENDING ladder, or ``None`` when unstated.

    Sorted here rather than at the reader, because ``EFFORT_ORDER`` is the one
    place the word order is defined and ``ModelSpec.reasoning_efforts`` is
    contractually ascending — ``next_effort`` indexes it and ``_lower_effort``
    walks it downward.

    Words outside ``EFFORT_ORDER`` are DROPPED rather than kept or passed
    through: ``resolve_effort`` calls ``EFFORT_ORDER.index`` on every rung of
    the ladder it is clamping toward (``effort.py:212``) and raises ValueError
    on an unknown one — on a failover hop, i.e. the request meant to rescue a
    turn. Dropping costs one rung on a model whose vocabulary grew; keeping
    costs the turn. A list that is ENTIRELY unknown words returns ``None``
    (unstated), not ``()``, so the table still answers.
    """

def _effort_default(value: object, ladder: tuple[str, ...] | None) -> str | None:
    """The listing's default rung, but only when it is ON the stated ladder.

    A default the ladder does not contain cannot be selected by ``/effort`` or
    reached by ``shift+tab``, and seeding it would put a level on the status
    band that the cycle can never return to. No row violates this today (0 of
    153); the guard is for the day one does.
    """
```

`_row_from_gemini_entry` and `_fetch_anthropic` are **not** changed: neither
wire carries the field (verified against the cached documents and the live
models.dev projection).

### 3.3 Cache round-trip — `_rows_from_payload` (`discovery.py:1256`)

The document is written by `dataclasses.asdict`, so a tuple lands as a JSON
list and `None` lands as `null`. The reader adds:

```python
                reasoning_efforts=_effort_ladder(entry.get("reasoning_efforts")),
                reasoning_default_effort=_effort_default(
                    entry.get("reasoning_default_effort"),
                    _effort_ladder(entry.get("reasoning_efforts")),
                ),
```

Reusing the same coercers is what makes the round-trip faithful: a stored
`null` reads back as `None` (silence preserved — the same trap
`supports_images` documents at `:1310-1314`), and a stored list re-sorts and
re-filters identically. Compute the ladder once and pass it to both.

### 3.4 Merge — `_merge_one` (`discovery.py:1135`)

```python
        # The listing's ladder is the router's own statement about the request
        # it will accept, and it wins when it speaks. The registry has no field
        # of its own here, so silence leaves ``None`` and the SPEC BUILDER falls
        # back to ``model.effort``'s table — the merge deliberately does not
        # consult the table itself, so a row stays a faithful record of what the
        # wire said and one function owns the fallback.
        reasoning_efforts=row.reasoning_efforts,
        reasoning_default_effort=row.reasoning_default_effort,
```

This is the smallest correct rule: `ModelInfo` has no ladder to merge against
(§3.5 keeps it that way), so there is nothing to fall back to at this layer.
Passing the row's value through unchanged is not a no-op — `_merge_one`
constructs a fresh `DiscoveredModel`, so omitting these fields would **silently
drop them to the default**, which is exactly the class of bug this file's
comments keep recording.

### 3.5 `ModelInfo` and `_info_from_discovery` (`configure.py:932`)

**Decision: do not add a field to `ModelInfo`.** `ModelInfo` is the *registry
row* type — a pydantic model with legacy consumers, duck-typed stand-ins in
tests, and a `description`/`id`/`name` contract. Nothing in the shipped registry
can state a ladder, so a field there would be permanently `None` for every
bundled row and would exist only as a transport between two functions in the
same module.

Instead, thread the row's ladder to the spec builder without laundering it
through the registry type. Two options:

* **(i) A parallel out-parameter.** `_info_from_discovery` also returns the
  matched `DiscoveredModel`; `_resolve_model_info_cached` stashes the ladder in
  a small per-(provider, model) memo that `build_model_spec` reads.
* **(ii) A private attribute on the returned `ModelInfo` copy.**

**Recommend (i)**, and specifically: give `resolve_model_info` a sibling
`resolve_effort_ladder(provider, model_id) -> tuple[tuple[str,...] | None, str | None]`
that reads the *same* memoized listing rows the resolver already holds, so it
costs no extra I/O. (ii) invites a pydantic private-attr that `model_copy(deep=True)`
(`configure.py:1365`) will not carry, which is a silent-loss bug of the exact
kind this subsystem keeps re-learning.

**The defensive read.** `build_model_spec` is handed a duck-typed `info` and
already defends at `configure.py:314-324` for `name`. The new read must be
equally defensive, but note it is *not* reading `info` at all under (i) — which
is itself the argument for (i). The spec builder becomes:

```python
    listing_levels, listing_default = _listing_effort(canonical, model_name)
    if listing_levels:
        effort_levels = listing_levels
        reasoning_effort = listing_default
    else:
        effort_levels = supported_efforts(model_name)
        reasoning_effort = default_effort(model_name)
```

`_listing_effort` must be **total and non-raising** — it sits on the session
start path and `build_model_spec` is called from a TUI repaint. Wrap it in the
same `except Exception` discipline the file already uses at `:1279` and
`:916`, returning `(None, None)`.

**Do not route this through `_fill_from_row`** (`configure.py:1015`). That
function is the second-hand-catalogue path and it *deliberately refuses to take
capabilities* — "`supports_images` is not taken at all: … a second-hand listing
has no standing to issue one" (`:1023-1026`). The OpenRouter ladder for an
`openrouter/*` id is first-hand about its own route; the same document consulted
for a *direct* provider's price under `OPENROUTER_NAMESPACE`
(`prices.py:191-201`) is not. Keeping the ladder out of `_fill_from_row` is what
prevents an OpenRouter ladder leaking onto `anthropic/claude-opus-5` resolved
through the direct provider. **This is the single most important boundary in the
design** and a reviewer should check it explicitly.

### 3.6 Capture versions (`discovery.py:162`)

```python
LISTING_CAPTURE_VERSIONS: dict[str, int] = {"anthropic": 2, "openrouter": 5, "radient": 5}
```

**Only `openrouter` and `radient` bump**, to 5. They are the only transports
whose parser learns a new field; `anthropic` stays at 2 and every default-1
transport stays at 1, which is the whole point of the per-transport map
(`:126-129`).

What a stale read does in the meantime: a version-4 document has a perfectly
valid shape in which every row's `reasoning_efforts` is absent → `None` →
"listing said nothing" → the table answers, i.e. **exactly today's behaviour**.
Nothing crashes and nothing lies; the fix would simply be invisible for up to
24h. That invisibility is precisely what the stamp exists to prevent
(`:143-151`), so the bump is required, not optional.

Cost, stated plainly because the module insists on it (`:136-141`): one
synchronous refetch of the OpenRouter listing (~700 KB), once per install, on
the first resolution after upgrade, on the calling path. Paid only by users with
an OpenRouter or Radient credential. Same bill as versions 2, 3 and 4.

The `LISTING_CAPTURE_VERSIONS` docstring needs a "Version 5" paragraph in the
established voice.

### 3.7 The dot/hyphen table gap (§0(a))

Fix it in the same PR, in `_EFFORT_TABLE`: the generation separator becomes a
character class (`claude-[a-z]+-4[.-]6(?!\d)` and likewise for the 4.7+/5+
arms). It is three characters per arm and it repairs 8 rows on *every* route,
including the offline path where no listing can help. Leaving it means the
listing masks a table bug that reappears the day OpenRouter is unreachable.

The `\d{2,3}`-not-`\d{2,}` reasoning at `effort.py:126-131` must be preserved:
a dotted separator must not let the 8-digit snapshot date back in. Worth a
regression test asserting `claude-opus-4-20250514` still returns `()`.

---

## 4. Q4 — Ordering and unknown words

**Normalisation belongs at ingest** — in `_effort_ladder`, i.e. in the parser
and the cache reader, before a row exists. Three reasons:

1. `DiscoveredModel` is consumed by the picker, the merge, and the spec builder.
   Normalising at the reader means normalising three times and disagreeing once.
2. The stored document then holds ascending ladders, so a cache round-trip is
   an identity rather than a re-sort.
3. It puts the `EFFORT_ORDER` dependency in one function, which is where the
   unknown-word filter has to live anyway.

All 153 rows arrive strictly descending today, but sort by
`EFFORT_ORDER.index` rather than reversing — a reverse is a bet on the wire's
ordering, and a sort is not.

**Unknown words are dropped** (rationale in §3.2). The alternative — extending
`EFFORT_ORDER` at runtime — is unavailable by construction: `EFFORT_ORDER` is a
module-level tuple whose *positions* encode the semantic ordering that
`resolve_effort`'s nearest-rung clamp depends on, and a word arriving from a
listing carries no information about where in that order it belongs. A new word
is a human decision, and the honest interim behaviour is to not offer a rung we
cannot rank.

**Additionally, harden `resolve_effort`** (`effort.py:209-213`) to skip ladder
members outside `EFFORT_ORDER` rather than raise. Ingest filtering makes this
unreachable via the listing path, but the function is public, is called with a
model id from `failover.py:1502`, and a `ValueError` there kills the rescue
turn. Defence in depth on a one-line guard is cheap.

---

## 5. Q5 — Blast radius

The change makes **91** more OpenRouter models carry a non-empty ladder.
Vendors affected: google 21, anthropic 8, meta 7, deepseek 7, thinkingmachines
6, z-ai 5, qwen 4, x-ai 4, nvidia 4, openai 4, and a tail.

**No model gains a seeded `reasoning_effort` on an aggregator route, and no
shipped registry row changes what it sends.** Two qualifications this sentence
once omitted, both established by later review rounds and stated exactly in
§5.1 and in the `configure.py` comment. First, on the DIRECT Anthropic route
the dotted spellings (`claude-opus-4.6` and friends) *do* newly seed `high`,
because the ladder repair gives them a ladder the table previously denied them
— reachable only by hand-typing a dotted id, and the route where Anthropic's
documented `high` ≡ omission equivalence genuinely applies. Second, the wire is
unchanged on the failover path only because §7 keeps it so: giving those ids a
ladder let a seeded value survive a hop onto an aggregator, which measured 18
live rows before `_carried_effort` closed it (F10). This section originally
sized the radius as "~99 models begin sending the key"; both halves of that are
now obsolete. Round-1 review recomputed the true figures
against the live listing — **137** models would have started sending the key
(not ~99, which counted ladder-gainers and missed 46 rows where the table
already gave a ladder but `default_effort()` returned `None`), **60** of them
`mandatory: true` (not 41) and **29** seeded at their top rung (not 25). Those
corrected numbers are recorded because they are what the seeding decision was
re-judged against; §5.1 then removed the seed entirely, so **the wire-key
radius is now zero**. What remains is the ladder radius: models that had no
ladder now have one.

| Consumer | Site | Behaviour change | Safe? |
|---|---|---|---|
| `_reasoning_effort` | `clients.py:634-652` | **No change.** The level starts unset on a listing-derived model, so the key stays omitted until the user picks a rung. §5.1 | Yes — the risk was removed, not accepted |
| `_effort_label` | `tui/app.py:20308-20339` | Band renders `auto` where it rendered `""` — the segment appears, which is the reported bug, and names the picked rung once one is chosen. | Yes |
| `next_effort` / shift+tab | `effort.py:216`, `tui/app.py:13448` | Cycling becomes available. Reads the spec, not the table — no second derivation. | Yes |
| `/effort` | `tui/app.py:13565+`, `:17515+` | Stops saying "not adjustable". | Yes — but §5.2 |
| `resolve_effort` | `failover.py:1502` | Re-derives from **the table**, not the spec — split brain. §5.3 | **No — must fix** |
| `map_tier_to_effort` / auto-effort | `effort_classifier.py:107-120`, `configure.py:3509` | Auto-effort begins acting on 91 more models. Off by default (`values.effort.auto`), so opt-in only. | Yes |
| `_effort_for` refit | `configure.py:1950-1965` | More models have ladders to re-fit onto; already guards membership. | Yes |
| `_lower_effort` (empty-truncation retreat) | `harness/loop.py:143-157` | Gains a retreat path on models that previously had none — a strict improvement. | Yes |
| `effort_ceiling` clamp | `harness/loop.py:896-905`, `configure.py:3534` | Both index the ladder only after `in ladder` checks. | Yes |
| `owned.set_effort` | `session/runtime/owned.py:700` | Phone can now set effort on these models. | Yes |
| `_ladder` / `_current_effort` | `mobile/tui_handle.py:1041-1053` | Reads the spec; already `try/except`. | Yes |
| `Session._lowest_effort` | `session/session.py:7303-7308` | **Errand/naming calls now clamp to the bottom rung on ~99 more models.** §5.4 | Mostly — check |
| `ModelSpec.reasoning` | `configure.py:365` → read only at `tui/app.py:20339` | Display only; one reader in the tree. | Yes |

### 5.1 The named risk: RESOLVED by measurement — the listing default is not seeded

**This section originally recommended seeding the listing's `default_effort`.
Round-1 QA measured the inference that recommendation rested on and it is
false. The decision is reversed; what follows is the evidence and the rule.**

#### What was proposed, and the inference under it

`_reasoning_effort` gates on `level in request.model.reasoning_efforts`. Before
this change `reasoning_efforts` is `()` for `google/gemini-3.8-flash`, so the
key is never sent. Seeding the listing default would have made
`body["reasoning_effort"] = "medium"` go out on every turn
(`clients.py:1173-1183`), and the argument for it was that this costs nothing:
OpenRouter's `default_effort` describes what happens when you send nothing, so
sending it explicitly *should* be a no-op. This note flagged that as "an
inference, not a documented guarantee" and asked QA to settle it.

#### What QA measured

Same prompt, alternating arms, `usage.completion_tokens_details.reasoning_tokens`:

```
z-ai/glm-5.3   ladder ('low','high','max')   seed 'max'   n=12 per arm
  WITH seed on the wire : median 200.0  mean 227.5  min 165  max 421
  key OMITTED           : median  92.5  mean  90.9  min  66  max 127
  explicit 'low'        : median  50.0
  -> median ratio 2.16x, distributions NON-OVERLAPPING (165 > 127)

google/gemini-3.8-flash  seed 'medium'  ->  351.5 vs 379.5  =  0.93x  (a no-op)
```

**The inference is false, and it is model-specific.** Sending the stated default
is a materially different request from omitting the key on `glm-5.3`, and
genuinely equivalent on `gemini-3.8-flash`. Because the router guarantees no
equivalence, a seeding rule cannot rely on one holding for any given model —
and `z-ai/glm-5.3`, `~z-ai/glm-latest`, `~z-ai/glm-flash-latest`,
`moonshotai/kimi-k3` and `~moonshotai/kimi-latest` were all seeded at `max`.

#### The consequence that actually decides it

The spend is small in absolute terms (~107 extra completion tokens ≈
$0.00047/turn at glm-5.3's rate). The disqualifying consequence is a
**correctness** one, and it follows directly from the same measurement:

> If omitting the key produces materially LESS reasoning than sending the
> stated default, then `default_effort` does not describe what omission
> actually does.

So painting it on the status band as the current level would be **a status band
asserting a depth that is not in force** — the one thing that segment must never
do, and the rule §6 already applies to the `reasoning` word. Seeding fails on
the band's own terms, not merely on cost.

QA also found the seed reaching rows whose listing says reasoning is off unless
asked: **15 models with `default_enabled: false` were seeded anyway**, measured
0 → 141 reasoning tokens. `mistralai/mistral-small-2603` is the sharp case —
ladder `('none','high')`, `default_effort: high`, so the seed moved it from off
to maximum. Not seeding resolves that population at the root; no
`default_enabled` subset check is needed, because nothing is seeded to gate.

#### The rule as shipped

**On an aggregator route, nothing seeds `reasoning_effort` — not the listing's
default and not the table's. On a direct provider route the table seeds exactly
as it always has.** The ladder and the seed are separate questions:

- **Ladder** — the listing wins where it speaks; the table answers its silence.
  Unchanged, and it is what fixes the reported bug.
- **Seed, aggregator (`openrouter`, `radient`)** — nothing, on every path.
- **Seed, direct** — the table's `default_effort()`, guarded by membership of
  the effective ladder so a narrowed ladder seeds nothing rather than clamping
  to a rung no source stated.

#### Why the table's default does not seed on an aggregator either

That seed rests on Anthropic documenting `effort: "high"` as exactly equivalent
to omitting the parameter — but that is a documented fact about **Anthropic's
own API**, and on an aggregator we are not talking to it. OpenRouter interposes
its own reasoning gate ahead of the upstream model, which it publishes as
`reasoning.default_enabled`, and that gate changes the answer. Measured, same
prompt, n=10 per arm, real 200s:

| model | key omitted | seeded `high` | `default_enabled` |
|---|---|---|---|
| `anthropic/claude-opus-4.6` | **0** reasoning tokens | **70** | `false` |
| `anthropic/claude-sonnet-4.6` | **0** | **164** | absent |
| `anthropic/claude-opus-4.5` | **0** | **190** | absent |
| `anthropic/claude-opus-5` | 65 | 67.5 | `true` |

So on that route sending `high` does not restate a default — it switches
reasoning **on**. That is the same defect as §5.1's `default_enabled: false`
finding, arriving through the table instead of through the listing. We therefore
assert the equivalence only on the route whose documentation establishes it.
Omitting is the one choice that needs no claim: it is what the user gets today
if they never touch the dial, and one keystroke sets a real rung.

#### What this costs, and warm/cold parity

**8 OpenRouter Anthropic rows that boot showing `high` today now boot showing
`auto`**: `claude-opus-5`, `claude-sonnet-5`, `claude-fable-5`,
`claude-fable-5.1` and their `:batch` twins. That is **wire-neutral** —
`opus-5` measured 65 vs 67.5 reasoning tokens, i.e. genuinely equivalent, and
the `fable-*` rows are `mandatory: true` so their reasoning is on regardless.
The band stops asserting a level we cannot substantiate on that route.

This also closes the warm/cold split completely. An earlier revision stopped
seeding only on the *listing* branch while the table branch kept its default,
which made the boot state of the 16 rows both sources answer for depend on
whether an HTTP call landed — `openrouter/anthropic/claude-opus-4.6` resolved
band `auto` with no key when the listing was reached and band `high` with
`reasoning_effort: 'high'` when it was not. Because neither arm seeds now, both
agree by construction. Verified over a live 424-row pull on both aggregators:
**0 rows** resolve a different seed warm than cold, and **0 rows** seed at all.

What this costs and what it keeps: an OpenRouter reasoning model boots showing
`auto` rather than a level. `auto` is an already-defined rung in this
vocabulary — every OpenAI reasoning model boots that way by design, and
`_effort_listing` renders it as a rung of its own — so this invents no new
display state. The reported bug is fixed by the **ladder**: the segment appears,
`shift+tab` and `/effort` cycle it, and the picked rung is shown.

**The wire delta, stated exactly.** On the aggregator route it is **zero**: the
~137-model listing-default blast radius this section was written to size is
gone, and no model there starts sending a `reasoning_effort` it was not already
sending. The dotted-id repair in `model.effort` (widening the separator to
`[.-]`) gives ~9 Anthropic ids the table ladder they should always have had, and
on that route they gain the ladder while sending no key at all.

On the **direct** Anthropic route those same dotted spellings do newly seed
`high`, and there that is safe for the reason the aggregator cannot borrow: it
is Anthropic's own API, which is what documents the equivalence. It reaches no
shipped registry row — all 18 are hyphenated, and every one was verified
byte-identical to base — so it is only reachable by typing a dotted id by hand,
which previously resolved no ladder at all.

`/effort auto` stays coherent and reads the same field it always did: on
Anthropic it restores the documented `high`; on a listing-derived model
`reasoning_default_effort` is `None`, so the key returns to being omitted and
the band returns to `auto`. Command and band agree by construction because both
read `spec.reasoning_default_effort` rather than re-deriving from the model
name — which is why that field is kept rather than removed.

### 5.2 `/effort auto` re-derives from the table

`tui/app.py:13581` and `:17527` both call
`default_effort(getattr(spec, "model_id", ""))` to restore "the model's own
default". After this change that returns `None` for a listing-derived model
whose default came from the listing — so `/effort auto` on
`google/gemini-3.8-flash` would report "the provider's default (nothing sent)"
while the band had been showing `medium`. **Both sites must read the seeded
default off the spec instead**, which argues for carrying it there.

**Still required after the §5.1 reversal, for a reason worth stating.** With no
listing seed, `reasoning_default_effort` is `None` on a listing-derived model,
so re-deriving from the table would now be wrong in the *opposite* direction:
the table would answer `high` for `openrouter/anthropic/claude-opus-4.6` and
`/effort auto` would restore a level the spec never seeded, putting the band
back into the state §5.1 rejected — naming a depth that is not in force. Reading
the one field keeps the command and the band in agreement by construction under
both rules, which is why the field is kept rather than removed.

Smallest fix: add `reasoning_default_effort: str | None` to `ModelSpec` beside
`reasoning_efforts` (`types.py:1568`), set once in `build_model_spec` from
whichever source won, and have both `/effort auto` sites read it. This also
removes the last model-name knowledge from the TUI, which
`effort.py:5-8` explicitly wants ("the TUI reads the RESULT off the spec (never
this table) so no widget carries model-name knowledge") — a rule these two sites
currently break.

### 5.3 `failover.resolve_effort` is a genuine split brain

`spec_for_target` (`failover.py:1490-1504`) builds `target_spec` via
`build_model_spec` — which will know the listing ladder — and then clamps with
`resolve_effort(model_id, …)`, which consults **only `_EFFORT_TABLE`**. Post
change these disagree: for `openai/gpt-5.4-pro` the spec offers
`medium/high/xhigh` while `resolve_effort` clamps against
`none/low/medium/high/xhigh` and can hand back `low` — a rung the spec says the
route rejects, written straight onto `reasoning_effort`. `_reasoning_effort`
would then drop it at the wire (it re-checks membership, `clients.py:650`), so
this is a *silent depth loss and a lying band*, not a 400. Still wrong.

**Fix: clamp against `target_spec.reasoning_efforts`, not the model id.** Add a
ladder-taking sibling to `effort.py` — `resolve_effort_in(levels, default,
requested)` — and let the existing `resolve_effort(model_id, …)` become a thin
wrapper over it for the offline/table path. That keeps one clamping algorithm
and makes the spec the single source of truth, which is the division
`effort.py:5-8` already claims to enforce.

**Qualification (F10), added after round 3.** "A hop clamps against the target
spec's ladder" describes the clamp but not *what is clamped*, and the missing
half is load-bearing: the value fed into the clamp is
`base.reasoning_effort`, which on a direct Anthropic base is the automatic
`high` **seed**, not a user's choice. Giving the dotted aggregator ids a ladder
(§3) is what let that seed survive the wire client's membership re-check, so
this section's own repair turned a hop into a second route to the wire that
§5.1's no-seed rule does not cover — measured through the real
`spec_for_target` and the real `clients._reasoning_effort` against the live
424-row listing at **18 `openrouter/anthropic/*` rows on head against 8 on
base**, with `openrouter/anthropic/claude-opus-5` sending nothing when selected
directly and `high` when reached via failover.

So the rule is: **a hop clamps against the target spec's ladder, and onto an
aggregator target it carries only a level the user actually CHOSE.** The choice
test is `reasoning_effort != reasoning_default_effort` — not a heuristic but the
documented contract of the field pair added in §5.2, which are equal at build
time and diverge exactly when the dial is moved. Direct→direct hops are
deliberately untouched (144 of 144 measured cells identical): an Anthropic seed
riding onto a direct OpenAI target is arguable on its own merits, but it
predates this change, and re-scoping it is a separate wire change on routes this
one does not otherwise touch.

One case is knowingly accepted as ambiguous: a user who explicitly picks the
level that *equals* the model's default produces a spec byte-identical to the
seeded one (verified by comparing `model_dump()`), so that choice is dropped on
a hop to an aggregator. Choosing the level the model already runs at asks for
the behaviour omission produces, so the target's own default honours the ask.
Separating the two would need a third `was_explicitly_set` field threaded
through every writer, the session rebuild and the `/model` switch — real cost on
every spec for a distinction that is unobservable on the direct route and that
the no-seed rule forbids acting on for an aggregator. If those states ever must
differ, that field is the fix; a sharper predicate cannot be.

### 5.4 `_lowest_effort` on errands

`Session._lowest_effort` (`session.py:7303-7308`) forces the **bottom rung** for
naming/errand calls, and its docstring reasons carefully about reasoning tokens
eating `ERRAND_MAX_TOKENS`. With ladders on 91 more models this now fires much
more often — which is the *intended* direction (it prevents exactly the silent
no-title failure it describes). One thing to check: 19 gainers have ladders
starting at `none`, so an errand on those will now send
`reasoning_effort: "none"`. Reassuring finding: **no OpenRouter row combines
`mandatory: true` with a `none` rung** (0 of 153), so this cannot ask a
mandatory-reasoning model to stop reasoning. Worth a QA cell anyway.

---

## 6. Q6 — The `reasoning` flag for the 144 ladder-less rows

**Decision: no. Leave `reasoning` derived as it is.**

The 144 rows carry `{"mandatory": …, "default_enabled": …, "supports_max_tokens": …}`
and advertise `reasoning`/`include_reasoning` but **not** `reasoning_effort`.
Splitting them: 78 are `mandatory: false` with no `default_enabled` at all, 31
are `mandatory: true` with no `default_enabled`, 26 are `(false, true)`, 7 are
`(false, false)`, 2 are `(true, true)`.

Arguments for setting `reasoning=True`: the band would say `reasoning` for
`deepseek`/`qwen` thinking models it currently renders blank, which is more
honest than nothing.

Arguments against, which win:

1. **The payoff is one word on the band, and `ModelSpec.reasoning` has exactly
   one reader in the whole tree** (`tui/app.py:20339`). Verified: no wire client,
   no loop, no failover path reads it. So this is a pure cosmetics change
   dressed as a capability change.
2. **The 7 `default_enabled: false` rows would be actively wrong.** The band
   would assert `reasoning` for a model whose listing says reasoning is off
   unless you ask.
3. **It expands scope into a second precedence question** (does the listing's
   `reasoning` flag overrule the name markers at `configure.py:292-294`?)
   without a reported bug behind it. The reported bug is the missing effort
   segment, and §1-§5 fix it.
4. The existing derivation already catches the loud cases via name markers
   (`reasoner`, `thinking`, `o1`, `o3`, `deep-research`).

**Do nothing here.** If someone later wants the band to say `reasoning` for
`qwen3.8-flash`, that is a separate, small, independently-reviewable change
whose merge rule is `default_enabled is not False`. It should not ride this PR.

---

## 7. Summary of changes

| File | Change |
|---|---|
| `model/discovery.py` | `DiscoveredModel.reasoning_efforts` (3-state) + `reasoning_default_effort`; `_effort_ladder` / `_effort_default` coercers; read in `_row_from_openai_entry`; round-trip in `_rows_from_payload`; pass through `_merge_one`; `LISTING_CAPTURE_VERSIONS` openrouter/radient → 5 + docstring paragraph |
| `model/effort.py` | Dot/hyphen fix in the three Anthropic arms; `resolve_effort_in(levels, default, requested)` extracted; `resolve_effort` hardened against unrankable rungs |
| `model/configure.py` | `_listing_effort(provider, model_id)` reading the memoized rows \u2014 the **ladder alone**, so a listing default cannot be seeded by accident (\u00a75.1); `build_model_spec` prefers it over the table; both constructors warm the memo (the discovery path and `_info_from_listing`, so the ladder is not a function of the caller); one bounded write enforces the memo ceiling; **no `_fill_from_row` change** |
| `harness/types.py` | `ModelSpec.reasoning_default_effort` |
| `providers/failover.py` | `spec_for_target` clamps against `target_spec.reasoning_efforts` — and carries only a user's CHOICE onto an aggregator target, never the automatic seed (`_carried_effort`) |
| `tui/app.py` | Two `/effort auto` sites read the spec's default, not `default_effort(model_id)` |

Not changed, deliberately: `ModelInfo`, `_fill_from_row`, `prices.py`,
`_row_from_gemini_entry`, `_fetch_anthropic`, `ModelSpec.reasoning`.

---

## 8. Risks a reviewer must check

1. **The wire key. — RESOLVED in round 1; the risk was removed, not accepted.**
   QA measured that sending the listing's stated default is not equivalent to
   omitting it (`z-ai/glm-5.3` at `max`, 2.16× the reasoning tokens, n=12,
   non-overlapping), so §5.1 no longer seeds it. **No model starts sending
   `reasoning_effort`.** What a re-reviewer should now confirm is the negative:
   that a listing-derived model boots with `reasoning_effort=None` and the key
   absent from the body, and that the TABLE-derived Anthropic seed (`high`,
   documented as identical to omission) still survives. §5.1
2. **The `_fill_from_row` boundary.** Confirm no path lets the OpenRouter
   ladder reach a model resolved through a *direct* provider via
   `OPENROUTER_NAMESPACE` (`prices.py:191`). This is the leak that would
   violate §2 in the wrong direction. §3.5
3. **`_merge_one` field-drop.** It rebuilds the dataclass; an omitted field
   silently defaults. Assert a listing ladder survives a merge with a registry
   row.
4. **Cache round-trip fidelity.** A stored `null` must read back `None`, not
   `()`. A `()` would become a denial and strip the table's answer — the
   `supports_images` trap at `discovery.py:1311-1315`, repeated.
5. **Capture bump scope.** Only openrouter/radient. An accidental global bump
   empties the model list for aggregator-only users (`:126-129`).
6. **The dot/hyphen regex.** `claude-opus-4-20250514` and
   `claude-sonnet-4-20250514` must still return `()` — the 8-digit-date
   swallow the `\d{2,3}` bound exists to prevent (`effort.py:126-131`).
7. **Narrowing is user-visible.** `openai/gpt-5.4-pro` loses `none`/`low` and
   `openai/gpt-5-pro` drops to a single rung. If a user had `none` selected and
   persisted, confirm the clamp path (§5.3) lands somewhere sane rather than
   inverting the intent — `resolve_effort`'s docstring
   (`effort.py:180-195`) is explicit that `none` must not become `high`.
8. **Offline parity.** With the listing unreachable, every model must resolve
   to exactly today's ladder. Test with the cache emptied and network refused.
