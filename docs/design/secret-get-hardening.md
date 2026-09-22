# Hardening `lop secret` against accidental agent credential leaks

**Status:** design record, with the programme's decisions in **§0** — what
shipped, what is deferred, and why. Written before the code; §0 was added after
the operator's ruling on the single decision this document escalated. Read §0
first if you want the state of the tree rather than the reasoning.

**Predecessors this extends, rather than replaces:** `docs/design/secret-store.md`
(the store, its four access surfaces, §6 redaction, §8 what each design stops, §9
residual risk, §13 compatibility and failure modes) and `guide://credentials`
(`local_operator/guides/credentials/GUIDE.md`, the agent-facing vocabulary).
`local_operator/redaction_shapes.py`'s module docstring carries the definition of
"compromised" and the two model-visible surfaces the shape pass does not reach;
both are quoted below rather than paraphrased.

**Evidence.** Every claim about current behaviour below carries a `file:line`.
Three of them I re-derived by reading the code and, where the claim was about
behaviour rather than structure, by driving the real functions — the probes and
their raw output are in Appendix A. Three of the handoff's claims did not survive
that: they are corrected in §1.3, and the corrections change what PR C has to do.

---

## 0. Decision record: what this programme implements, and what is deferred

Recorded after the design above was accepted in part. **Read this section first**
if you want the state of the tree rather than the reasoning behind it; it also
names the one decision that is the operator's to make.

### 0.1 R1 — the `get` default — is DEFERRED, not rejected

The recommendation in §2.3 stands as written: A1 (the harness supplies the bytes
for a proven consumer sink, so that `get` can become value-or-nothing) is right
about the *mechanism*. The ruling is that it must not ship on a manager's or a
coder's authority, because it **deliberately breaks a documented, load-bearing
contract for every caller, not only for the harness path**:

* `secrets/cli.py:1-38` states the stdout contract as *the* hard requirement of
  the CLI half — the exact stored bytes, no trailing newline — because "one
  banner line, one ANSI colour code, one progress note on stdout and every
  consumer silently receives a corrupted credential — which fails as a confusing
  401 from a remote service, not as an error anyone traces back to here";
* the pre-execution scan cannot see a value retrieved **inside a script the agent
  merely invokes** (`bash scripts/deploy.sh`, where the script interpolates `lop
  secret get`). PR #1429 records this as one of its own gaps, so a change to
  `get`'s default would reach callers the harness will never scan;
* it would change the meaning of the operator's own documented forms in
  `guides/credentials/GUIDE.md` — `lop secret get NAME > /tmp/token`
  (`GUIDE.md:50-54`) and `lop secret get NAME | shasum -a 256`
  (`GUIDE.md:120-123`) — and the brief for this programme requires the documented
  consumer form to keep working.

Changing a CLI contract that the operator's documentation and tooling depend on is
the operator's decision. **The default-changing half therefore ships only on their
word**, and the recommendation is preserved in full — §2.2's option table and
§2.3's semantics (exit 3, the refusal text, the `describe` descriptor, the audit
events of §2.3(5)) — so a follow-up can start from it without re-deriving
anything. The constraint that binds any opt-in, whichever shape it takes, and the
reason A1 was chosen over a flag or an environment variable, is unchanged: **it
must not be silently settable by the agent in the same call it uses.**

### 0.2 What ships now, and the invariant that holds it together

| workstream | branch | PR | what it does | what it does not |
|---|---|---|---|---|
| **B** — pre-execution refusal | `feat/secret-sink-scan` | **#1429** | §3, as designed: keys on the data FLOW (a secret-bearing source reaching a printing sink) and refuses the `bash`/`eval` call before any child exists, naming the rule, the span and the rewrite | §3.4's tainted-path ledger (scoped out, recorded as its own gap), a script invoked by name, flow across calls |
| **C** — the scrubber's deterministic holes | `fix/redaction-transform-hardening` | **#1428** | §5.2's transform normalisation (wider family list, 12-character floor) and the eval worker's streamed frames | the multi-line release-point defect — measured still live on its head; see the amendment in §5.1 |
| **A** (additive) | `feat/secret-identity-surface` | **#1430** | `lop secret describe NAME --length --fingerprint` (value-free identity; equal values fingerprint equal) and `lop secret get NAME --reveal`, which prints bytes only when stdin and stdout are both a terminal, audited as `reveal` | **R1** — `get`'s default is unchanged (0.1) |

**The invariant this programme keeps: `get`'s byte contract is unchanged** —
exact stored bytes, no trailing newline, `rc 2` with empty stdout on failure, and
its ordinary `get`/`ok` audit row. Every control above is additive beside that
contract, or a refusal standing in front of it; none of them alters it.

### 0.3 D is not built, and this is the reason

One line: **the classification layer's own contract forbids it** — "Nothing here
may gate a capability, change an approval tier or alter a tool's availability"
and "No history, no compaction summary, **no tool results**"
(`classification/__init__.py:20-28`) are precisely the two things a pre-execution
guard is. §4 carries the full reasoning, including the constraints a purpose-built
seam would have to satisfy. It is revisited only on evidence that the
deterministic layer's residual is material — the measured count §4 names — not on
the argument that a model might be more accurate.

## 1. The problem as I found it

### 1.1 What happened

In a Minerva QA session an agent ran

```bash
for k in A B C D; do v=$(lop secret get "$k") && echo "$k = $v" || echo "$k = MISSING"; done
```

and the values reached the model's context. `guide://credentials` names this exact
form as forbidden (`GUIDE.md:37-43`, "the transcript is the one place a value must
never land, because the transcript is the model's context"), and the harness said
so — the incident notice (`local_operator/incidents.py:1027`, the
`[credential redaction] rotate it — …` text) told the operator to rotate.

Then, to read a hostname the mask kept replacing, the agent printed it reversed
(`moc.avrenimog.aq.ppa-aq`). That is a deliberate bypass of a control, and it
defines the threat class this work must handle: not only the accident, but the
*shaped* re-spelling of a value to get past a filter.

### 1.2 What the code does today, in its own terms

* `lop secret get NAME` writes the exact stored bytes to `sys.stdout.buffer` and
  adds no newline (`local_operator/secrets/handlers.py:239-259`), routed through
  `local_operator/secrets/access.py:299` (`retrieve_secret`) so the §6 notice
  fires and is acknowledged before those bytes exist.
* **The stdout contract is load-bearing and deliberate.** `local_operator/secrets/cli.py:1-38`
  states it as "the hard requirement of that half is stdout purity on `get`", with
  the reason: "one banner line, one ANSI colour code, one progress note on stdout
  and every consumer silently receives a corrupted credential — which fails as a
  confusing 401 from a remote service, not as an error anyone traces back to
  here." The guide repeats the consumer half at `GUIDE.md:45-48`: non-zero exit
  with **empty stdout** on any failure, which is what makes `$( )` safe.
  I verified the bytes and the absence of a trailing newline against a real
  isolated store (Appendix A.1): 27 bytes, `xxd` shows no `0a`.
* There is **no guard, only a filter**. Nothing in the tree refuses a printing
  construct before it runs. What exists is
  `local_operator/secrets/runtime.py:93-141`'s process-wide `_RedactionLedger`
  (registered values, longest first, `str.replace`), the exact-value half of
  `local_operator/variables.py:491-556` (`redact`/`redact_with_report`), the
  spelling-based shape pass (`local_operator/redaction_shapes.py`), and the
  per-chunk pipe filter `local_operator/tools/builtin.py:2150` (`_PipeRedactor`).
* The agent's other surfaces already refuse to carry a value:
  `handlers._list` (`handlers.py:287-293`) has no flag that prints values, by
  design ("a `--values` flag here would turn a single audit entry into a bulk
  export"); `tools/secret_tool.py:228-243` answers `op="retrieve"` with
  `describe`, not with bytes, "misreporting the one trail the operator relies on"
  being the stated reason; `store.py:118-134`'s `SecretRecord` is value-free by
  construction ("there is no field here to print").
* The remainder of this document calls things `get`, and `get`'s stdout is the
  only value-returning surface an agent has.

### 1.3 Three corrections to the handoff, because they change the work

None of these invalidate the programme; two of them move work between PRs and one
of them changes PR C's implementation.

**(a) "Chunk boundaries: a value split across two chunks is in neither" — not
true in general.** `_PipeRedactor` holds back a lookbehind sized from the longest
known value (`builtin.py:2207-2210`), re-reads the value set per chunk
(`builtin.py:2211-2223`, called at `builtin.py:2901-2904`), and
`_release_point` (`builtin.py:2325-2363`) explicitly refuses to cut through a
known value — a loop that pulls the cut back to the value's start
(`builtin.py:2351-2362`, comment: "Never cut through a KNOWN value"). Probe A.2
confirms it for a single-line value split mid-value across two writes: released
`b'prefix [redacted] suffix\n'`.

**The hole is narrower than "chunk boundaries" and more embarrassing than that:
it is every value containing a newline.** The cut is placed after the last
newline in the accumulated text (`builtin.py:2333`), and the "never cut through a
known value" loop can only see a value that is *fully present* — `text.find(secret)`
returns `-1` while the value's tail has not arrived. So the prefix before that
newline is released raw, and the remainder is released raw in the following
chunks; by the time `feed(b"", final=True)` runs, `pending` holds only the tail
and no `replace` can match anything. Probe A.2 reproduces it:

```
[multi-line split at newline]  secret='line-one-aaaa\nline-two-bbbb'
  chunks=[b'out: line-one-aaaa\n', b'line-two-bbbb\n']
  released=b'out: line-one-aaaa\nline-two-bbbb\n'
  RAW-BYTE LEAK: True
```

This is not a rare shape. The store's own file-shaped secrets — a service-account
JSON, a PEM — are multi-line by definition, and `lop secret set --from-file`
(`GUIDE.md:106-123`) and `DEFAULT_FILE_ENV_VAR` (`cli.py:41-46`) exist precisely
for them. What the leaked bytes reach is scoped in §5.3 and §6.4; the settled
tool-result path still masks a registered value, so the leak is to the live
surfaces and the retention/spill buffer (`builtin.py:2905-2917`, `sink.append(safe)`
— the spill holds filter *output*, so a filter leak is a spill leak).

**(b) "Transformations pass through untouched" — true of the exact-value pass,
only half true of the pipeline.** Probe A.3, over a value the session knows
(retrieved through the seam):

| form | exact + shape pass | shape pass alone |
|---|---|---|
| verbatim, bare | masked (exact) | not masked |
| verbatim, assignment `NAME=` | masked (exact) | masked |
| **reversed, bare** | **not masked** | **not masked** |
| reversed, assignment | masked (shape) | masked (shape) |
| **base64, bare** | **not masked** | **not masked** |
| base64, assignment | masked (shape) | masked (shape) |
| **hex, bare** | **not masked** | **not masked** |
| **per-char interleave, bare** | **not masked** | **not masked** |
| **piecewise, bare** | **not masked** | **not masked** |

The shape pass catches a *transformed* value only when the transform happens to
sit in a spelling the table recognises (`credential-assignment`), which is a
coincidence of position rather than coverage. The incident's aggravating act —
a value printed reversed — is in the "not masked" rows unless it was also
assignment-spelled. This is the case PR C's transform normalisation exists for,
and it is worth being precise about: it is a *detection* control, not a barrier.

**(c) The broker is not an enforcement point in the default tier, so PR A's gate
cannot live only there.** `access.py:360-388` degrades on `BrokerUnavailable`
(availability, not policy, §13) but also — and this is the part that matters —
falls through to `open_store(base).get(...)` on `BrokerDenied` unless the tier is
`passphrase`. Verified live against an isolated default-tier store: the audit log
shows `deny:retrieve` immediately followed by `get ok` for the same pid
(Appendix A.1). `docs/design/secret-store.md` §8 states the same conclusion for
the same reason ("a denial there is a fallback, not a refusal").

Consequence, and it is a design constraint rather than a nitpick: **any refusal
this programme adds must be enforced in the process that produces the bytes**
(the CLI handler) with the broker as a second, notifying layer. A gate that only
the broker applies is bypassed by the store's own documented fallback the first
time the broker is not there to answer.

**(d) The §6 notice is not what failed.** `retrieve_secret` announces and waits
for the ack *before* the value exists (`access.py:299-330`, and the broker's
denial is "believed" rather than degraded, `access.py:366-372`), so the notice
firing in the incident means the ordering worked and the agent read a notice that
said "rotate it". The gap is that nothing stopped the *printing construct* from
being written in the first place, and no filter can, because a filter sees the
value only after the decision to print it has already been made — and, per (a)
and (b), only in the forms it recognises.

---

## 2. The central decision: what `get` defaults to, and what "the opt-in" is

### 2.1 The two shapes really are indistinguishable at the CLI

The brief's two constraints collide:

* **R1** — `lop secret get NAME` defaults to an identity-preserving descriptor;
  raw bytes require an explicit, gate-able, audited opt-in.
* **R2** — `curl -H "Authorization: Bearer $(lop secret get X)" …` keeps working
  exactly as it does today, including exact bytes and no trailing newline.

At the CLI these are the same shape. In both, the process's stdout is a pipe to a
shell (`$( )` is a pipe in both cases), the environment is the same, the ancestry
is the same, and the argv is the same. Nothing the CLI can observe distinguishes
"a substitution feeding a consumer's argv" from "a substitution feeding `echo`".
I state this as a fact about the primitives, not as a limitation to be worked
around: the information the decision needs — *what the substitution's destination
is* — exists only in the **command text**, which the harness holds and the CLI
does not.

So the decision has to be made by the layer that can see the sink, and the CLI has
to be told what that layer decided. Every option below is a different answer to
"told how, by whom, and what does the CLI do when it has not been told".

### 2.2 The options

**A1 — harness supplies the bytes (recommended).** The pre-execution scan (PR B)
runs over the command text in `execute_bash` (`builtin.py:2589`) before the
spawn. Where it proves that every `$(lop secret get NAME)` occurrence flows into
a consumer sink, the harness **retrieves the value itself, in the session
process**, injects it into the child's environment under a per-secret variable
(the mechanism session credentials already use: `builtin.py:2624-2633` builds
`injections` from `store.credential_env()`, design §5.4), and rewrites the
occurrence to that variable — `"$(lop secret get X)"` becomes `"$LOP_SECRET_X"`,
unquoted becomes unquoted, so word-splitting behaviour is preserved. An
occurrence the scan cannot prove is left alone, and `get` then answers with
nothing (§2.3).

**A2 — sink-aware capability.** Same seam, same verdict, but the bytes stay
produced by the CLI: the harness mints a per-call, single-use, nonce bound to the
minting process tree and its TTL, the CLI redeems it to be allowed to write raw
bytes.

**A3 — required per-call flag** (`lop secret get NAME --reveal`), with no
harness involvement.

**A4 — approval-gated reveal only**: raw bytes need an interactive approval via
`local_operator/harness/approval.py:105-120` (`ask_approval`); non-interactive
fails closed.

**A5 — keep `get` exactly as it is**, re-point the guide's recommended consumer
form at `lop secret run`/`lop secret file`, and let B (refusal of printing
constructs) + C (filter hardening) carry the whole programme; the descriptor moves
to `describe` and the reveal becomes human-only and gated.

| | (a) R1 + R2 both hold | (b) agent cannot set the opt-in in the same call | (c) non-interactive with nobody to ask | (d) covers what B is blind to |
|---|---|---|---|---|
| **A1** harness supplies bytes | **yes** — R1 holds *because* `get` no longer needs to be the reveal path; R2 holds byte-exact, because the harness puts the same bytes in the same argv position | **yes** — the opt-in is a rewrite the harness makes; the model's text cannot request it, and a `$LOP_SECRET_X` written by the model in a later call expands to nothing (the variable exists only in that child) | **yes, and this is its strength** — the sanctioned form needs no approver at all; the uncertain cases are *refused* with a rewrite hint | **yes** — every unapproved `get` (inside `eval`, an opaque script, a Makefile, a base64-encoded command, a `python -c`) hits the CLI default, not the scan |
| **A2** capability | yes | yes, if the nonce is minted only on a favourable verdict and bound to the minting tree; a bare env var is **no** — `LOP_REVEAL=1 lop secret get X` is the agent setting it in the call it uses | partially — needs the broker to answer, and by §1.3(c) the default tier degrades | yes |
| **A3** flag only | **no** — R2 breaks: the documented form has no flag and would lose the bytes | **no** — same call, same text | yes (fails closed) | yes |
| **A4** approval only | **no** — R2 breaks in every unattended run, which is most of them | yes (strongest) | fails closed, which here means the sanctioned form stops working | yes |
| **A5** leave `get` alone | yes (nothing changes) | n/a | n/a | **no** — the leak stays open wherever B has no rule: transforms, another tool, `cat` of a redirected file, a value re-read later |

The table's decisive column is the last one. The point of R1 was never elegance:
it is that **the CLI's default is the fail-safe direction for a scan that is
wrong**. Under A1/A2/A3/A4 a mis-judged sink costs a broken request, and the value
stays put; under A5 a mis-judged sink is a leak. A1 and A2 are equivalent in that
respect and differ in machinery, so the simpler one wins.

### 2.3 The recommendation, precisely

**Recommendation: A1, with A5's explicit refusal as the CLI's default behaviour
and A4's interactive prompt as the human path. Do not put the descriptor on
`get`'s stdout.**

That last clause is the one place this document departs from the PR-A brief, and
the reason is the brief's own constraint. A descriptor printed on stdout is
*non-value* bytes delivered into a pipe (or a file, or a consumer's argv) that
asked for a value. There is no way to deliver it there without risking exactly the
failure `cli.py:16-27` was written to prevent — "every consumer silently receives
a corrupted credential — which fails as a confusing 401", except that here the
credential is silently *replaced* rather than corrupted, and the operator learns
nothing. A descriptor belongs on a surface whose contract is identity, and the
store already has one. So:

**(1) `lop secret get NAME` — value or nothing.**

| caller state | stdout | exit | stderr | audit event |
|---|---|---|---|---|
| the harness supplied the bytes for this call (A1) → in that call the CLI is not reached at all; the substitution was rewritten | n/a | n/a | n/a | the harness records `get-for-consumer` (see (5)) |
| a human at a TTY asks for a reveal (`--reveal`) | exact bytes, no newline | 0 | — | `reveal` outcome `tty` |
| `--reveal` with no TTY and no harness authorisation | empty | 3 | why, and the sanctioned forms | `reveal` outcome `refused` |
| no `--reveal`, no harness authorisation (any caller) | empty | 3 | why, and the sanctioned forms | `get` outcome `refused` |

Exit `3` is new and distinct from argparse's `2` and the store's failure codes, so
a script can tell "you are using a form this store will not print" from "the
secret is missing" and from "the store is damaged". The stdout invariant that
`GUIDE.md:45-48` documents — non-zero with empty stdout — is *preserved* rather
than changed, and the refusal text is actionable:

```
lop secret get: refusing to print GITHUB_TOKEN to stdout.
  To USE it:  lop secret run --secret GITHUB_TOKEN -- <command…>
              lop secret file NAME -- <command…>            (file-shaped secrets)
              curl -H "Authorization: Bearer $(lop secret get GITHUB_TOKEN)" …   (see note)
  To IDENTIFY it: lop secret describe GITHUB_TOKEN
  To see it as a human: lop secret get GITHUB_TOKEN --reveal    (asks at a TTY)
```

with the note explaining that the `$( )` form is honoured only when the harness
can prove the sink, which is the honest thing to say and doubles as the model's
instruction to re-spell the command into a shape the scan can approve.

**(2) The descriptor lives on `describe`, extended.** `lop secret describe NAME`
(`handlers.py:336-349`) already prints name / id / kind / description / timestamps
and never a value. Extend it with the two fields that make it an *identity*:

```
name         GITHUB_TOKEN
kind         string
length       40 bytes
fingerprint  hmac-sha256:4f9c…        (keyed; stable; not reversible)
```

* The fingerprint is `HMAC-SHA256(master_key, value)`, computed on demand in
  `describe` — no schema migration. Note honestly what it is: a keyed digest is a
  *confirmation oracle* for someone who already holds the master key, and such a
  holder can decrypt the store anyway, so it gives nothing away that the key does
  not. It is keyed rather than a bare `SHA256(value)` so that a low-entropy value
  cannot be brute-forced from the descriptor, and so that the descriptor is not a
  searchable artifact on disk.
* `length` requires the plaintext, and `describe` already decrypts it: name and
  description live *inside* the ciphertext (`store.py:203-217` `_payload`,
  `store.py:540-553`), so `describe` is a decrypting call today. Worth knowing
  before anyone claims the descriptor is "the path that never decrypts" — a
  `value_len` column would be needed to make that true, and I recommend *not*
  adding one until something asks for it.
* The same three fields are what the `secret` tool's `describe` op
  (`tools/secret_tool.py:244-256`) and the harness's `retrieve` receipt
  (`tools/secret_tool.py:228-243`) should carry, so the agent's "what is this
  secret?" surface and the operator's agree.

**(3) What the opt-in IS, and who can set it.**

* **`--reveal` at a TTY.** A human at a terminal is the case `docs/design/secret-store.md`
  §13 already treats as authorised (residual 4: "the terminal you unlock in is
  authorized for as long as it lives"), and the idiom is already in the tree:
  `handlers._remove` (`handlers.py:352-358`) asks at a TTY and requires an
  explicit flag when there is none. `--reveal` mirrors it: a prompt, not a flag
  that means "just do it", and an audit row either way. A non-TTY caller is
  refused, which is (c) — fail closed with nobody to ask.
  *Stated weakness, since it is real:* a pty is also what the `console` tool
  drives, so an agent can reach a genuine TTY by typing into a console surface.
  That is a deliberate, visible act (the console tab is on the operator's screen)
  rather than a silent one, and it is audited as `reveal` with the pid and
  executable, so it is attributable after the fact. It is named again in §6.8.
* **The harness's rewrite (A1).** Not a setting the model can touch: it is the
  scan's verdict applied by the session process. The model's command text cannot
  request it, grant it to itself, or carry it to a later call, and the variable
  it produces is valid only inside the child that call spawned. This is the same
  principle the tree already ships for gate loosening —
  `local_operator/harness/approval.py:132-140`: *"a model tool's own file write
  (the party being gated is not the authority that may lower its own gate — the
  reason this predicate exists)"* — applied to the credential path.
* **What is deliberately NOT an opt-in:** an environment variable, a config key,
  or a `--force`. Each is either writable by the model (`LOP_REVEAL=1 lop secret
  get X` is one call, and the constraint in the brief is exactly that this must
  not work), or it is a second approval prompt on a tool whose design says there
  must not be one (`builtin.py:2616-2621`: "A second gate here made the user
  answer twice per action, with the tier name rendered as the tool").

**(4) Denial is possible, and is the normal answer for an unapproved call.** Two
denials exist and both are visible: the scan's refusal (a tool result carrying the
rule label and the rewrite hint, no value, no prompt) and the CLI's exit-3 refusal.
A denial never degrades to an unnotified silent read, because the gate is in
`handlers._get` — the process that writes the bytes — not only in the broker
(§1.3(c)).

**(5) The audit trail, which is where "distinguishable from a normal `get`" is
settled.** The `audit` table's `event` column is free text (`store.py:105-109`)
inside the hash chain (`local_operator/secrets/audit.py:92-190`), and today a
retrieval is `event="get"` (`store.py:742`). The programme adds:

| event | outcome | written when |
|---|---|---|
| `get` | `ok` | descriptor/metadata read — i.e. `describe`; **not** a value read |
| `get` | `refused` | an unapproved `get` was asked to print a value |
| `reveal` | `tty` | a TTY reveal produced raw bytes on stdout |
| `reveal` | `refused` | `--reveal` with no TTY and no authorisation |
| `get-for-consumer` | `ok` | the harness retrieved on the child's behalf and injected it (A1) |

`get-for-consumer` is the row that closes the §6 case-2 blind spot, and it is the
one worth arguing for: today the `$()` form's retrieval is audited from the
*child's* pid, and the session only learns of it through the broker's
notify-and-ack dance (`access.py:299-330`). Under A1 the retrieval happens in the
session process, so the row names the session, the value is registered for
redaction directly (§6 case 1 instead of case 2), the notice fires on the
session's own path, and the audit says *why* the bytes were handed out. Row
shapes stay value-free (`audit.py:20-26`: "Never record a value").

### 2.4 What the guide becomes

`guide://credentials`'s consumer section (`GUIDE.md:26-67`) gains one precedence
rule, because the non-revealing forms are strictly better and already exist:

1. `lop secret run --secret NAME -- <command…>` and
   `lop secret file NAME -- <command…>` (design §7) — the value never crosses
   stdout, needs no scan verdict and no reveal, and cannot be read by the model.
   These become the *primary* recommendation for a consumer that takes env/argv.
2. `$(lop secret get NAME)` interpolated into the consuming command stays
   documented and stays working — it is short, it is in existing runbooks, and
   under A1 it is mechanised. The guide should say what makes it work: the sink
   has to be a consumer, and a printing construct is refused before it runs.
3. The "Do not do either of these" block (`GUIDE.md:37-43`) stays verbatim, with
   one sentence added: the store now *refuses* rather than warns, and the refusal
   says what to write instead.

---

## 3. PR B: the pre-execution scan, its seam, and its rule table

### 3.1 Where the scan runs, and why there

**In the tool, beside argument validation, before the spawn — not at the loop,
and not as an approval.** `execute_bash` (`builtin.py:2589`) already refuses
malformed calls inline (`builtin.py:2604-2610`) and already has a pre-abort
branch (`builtin.py:2612-2621`); the scan is the same kind of thing: a validation
refusal that returns an error `ToolResult` before any process exists. The
justification for *that* seam rather than the loop's:

* the loop does not know the tool's semantics; the rule table's vocabulary is
  per-language (`sh` command text vs a Python cell), and only the tool knows
  which it is holding;
* `execute_bash`'s deliberate lack of a second approval gate
  (`builtin.py:2616-2621`) means the scan must not be implemented as an approval
  question — it refuses, it never prompts;
* one seam covers every harness that runs a command: the interactive TUI, `lop
  exec`, headless runs, forks and subagents all reach the same tool, whereas a
  loop-level check would have to be threaded into each entry point. (`exec_mode.py`
  and `exec_session.py` start sessions rather than commands, so they are covered
  by construction.)
* the eval surface is a *different* tool with a *different* language:
  `execute_eval` (`local_operator/tools/eval.py:1055`) scans the cell with the
  Python rules.

**Module.** A new `local_operator/harness/secret_sinks.py`, stdlib-only, no
package imports beyond typing/dataclasses/re. Reasons for that home: it is policy
about how the harness runs model-authored text, it sits beside
`harness/redaction.py` (its sibling in the same story: that module publishes *which
tool's* bytes are being redacted) and `harness/approval.py`; and it must be
importable from both `tools/builtin.py` and `tools/eval.py` without dragging in
the crypto stack, the same constraint `secrets/cli.py:11-15` documents for the
CLI half.

**Interface** (pure functions, no I/O, no state):

```python
@dataclass(frozen=True)
class Finding:
    rule: str            # the rule label, quoted verbatim in the refusal
    span: tuple[int, int]  # where in the command text
    reason: str          # one sentence, model-facing
    rewrite: str | None  # what to write instead, when there is one

@dataclass(frozen=True)
class ScanVerdict:
    sources: tuple[Source,...]      # every value source found, with its sink verdict
    findings: tuple[Finding,...]    # refusals
    supplies: tuple[Supply,...]     # (name, variable, text-span) the harness may inject — A1
    unresolved: bool                # a source exists that the lexer could not place

def scan_shell(command: str) -> ScanVerdict: ...
def scan_python(source: str) -> ScanVerdict: ...
```

`scan_*` never raises: a fault becomes `unresolved=True` plus one finding.

### 3.2 The rule table: declarative, inspectable, testable one at a time

The rules are a **literal tuple of frozen dataclasses** in one place, not a
handful of regexes sprinkled through the tool:

```python
@dataclass(frozen=True)
class Rule:
    label: str              # stable id: appears in the refusal, the tests, the doc
    lang: Literal["shell", "python"]
    question: str           # the question this rule answers, in one sentence
    verdict: Literal["refuse", "supply", "allow", "neutral"]
    why: str                # the constraint it enforces, with the incident or non-finding
    examples: tuple[str, ...]        # must fire
    counterexamples: tuple[str, ...] # must NOT fire — the four non-findings live here

RULES: tuple[Rule, ...] = (...)
```

Inspection and per-rule testing follow from that shape, and are the reason for it:

* one test iterates `RULES` and asserts every `examples` entry produces that
  rule's verdict and every `counterexamples` entry produces none — a reviewer can
  read one rule's block and check it without holding the whole table;
* a test asserts labels are unique and every rule fires on at least one example
  (an unreachable rule is dead policy that will drift);
* a test asserts the module imports no package module (`tests/unit/test_import_graph.py`
  has the precedent), and one asserts the pair
  `("bash", "lop secret get")` never yields `allow` with `supplies` empty;
* the module itself is the inspection surface (`python -c "from
  local_operator.harness.secret_sinks import RULES; print(len(RULES))"`, plus a
  test that renders the table). I recommend **against** a new `lop secret rules`
  verb: `AGENTS.md`'s footprint ladder puts a CLI verb at rung 2 and this needs
  no new surface.

**The lexer is the real work, and it is the honest cost of PR B.** The
non-finding (iv) — a heredoc that merely *writes* a script containing the literal
pattern — cannot be handled by a regex over the text: `cat <<'EOF'` does not
expand and `cat <<EOF` does, and single-quoted strings, comments and `$( )`
nesting behave differently again. So the scanner needs a small shell tokenizer
that classifies regions as: command substitution (executed), double-quoted
(expanded), single-quoted (literal), heredoc body with a quoted delimiter
(literal), heredoc body unquoted (expanded), comment, assignment, command word,
argument position. "A value source inside a literal region is not a source" is
then a property of the region classification rather than a pattern, which is what
makes (iv) a *counterexample* instead of a special case.

### 3.3 Sources, sinks, and the four required non-findings

**Sources** (the gate condition — the scan is a no-op on text with none of these):

* shell: `lop secret get NAME` (with or without `$( )`), `lop secret file NAME`,
  a variable already tainted by either in this command (`v=$(lop secret get X)`;
  the A1-supplied `$LOP_SECRET_*` namespace), and the session-credential
  namespace (`$NAME` for names the session injected — the same values
  `builtin.py:2624-2633` puts in the child environment);
* eval: `secrets["NAME"]` / `secrets.get(...)`, and a name assigned from one.

**Sinks that refuse** (each is a rule):

| sink | why |
|---|---|
| `echo` / `printf` / zsh `print`, bare or as the consumer of a substitution | the incident's Case A; the guide's explicit prohibition |
| a bare `lop secret get NAME` in command or statement position | its stdout *is* the tool result; this is the incident's shape with no `echo` to blame |
| `cat` / `tee` / `head` … of a tainted variable, or of a path in the tainted-path ledger | §3.4 |
| `set -x` (or `bash -x`, `sh -x`, a `PS4` assignment) in a command containing a source | under xtrace every expansion is echoed to stderr, so the sink is the shell itself |
| `ps -ww` / `ps -ef` / a read of `/proc/*/cmdline` in a command containing a source | a consumer's argv is readable by any same-uid process — the store's own §2.2 rejects argv as a value channel for exactly this reason |
| python: `print`, `sys.stdout.write`/`sys.stderr.write`, `logging.*`, a returned expression that interpolates one | same, on the eval surface |
| python: `open(path, "w")` / `.write()` of one | creates the debt in §3.4; refuse only when combined with a *later read in the same cell*, else `allow`+ledger |

**Verdicts that allow, with the required non-findings as counterexamples:**

* **(i) `v=$(lop secret get X)` then use in a curl header / client argv** — the
  sanctioned form. Under A1 this is a `supply`: the harness injects and rewrites
  the assignment (`v=$LOP_SECRET_X`), which preserves unquoted word-splitting.
  Case B — a heredoc writing a script that *contains* the pattern — fires
  nothing, because the source sits in a literal region.
* **(ii) `${#VAR}` and `lop secret get X | wc -c`** — `allow`. Length-only sinks
  are an explicit allowlist (`wc -c`, `wc -m`, `shasum -a 256`, `sha256sum`,
  `md5`-family, `cmp`), and there is shipped precedent for treating a length as
  safe: `handlers._set` prints `len(value)` deliberately (`handlers.py:271-274`).
  Keep the list tight: a hash is fine, a `diff` is not (its output is bytes of
  the subject).
* **(iii) `lop secret list`, `lop secret get --help`, `lop secret describe`** —
  no source at all: the source pattern is the *value-returning* verb only. Note
  `--help` short-circuits before `handlers`, and `list`/`describe` never decrypt
  a value into output (§1.2).
* **(iv) a heredoc containing the pattern whose write target is a file** — no
  source, per §3.2's region classification. The `supply` rewrite must also refuse
  to touch text inside such a region, which is the same property read in the
  other direction: this is the single highest-risk false-positive class in the
  whole scanner (agents write scripts as heredocs constantly), and the test corpus
  should carry several spellings of it.

### 3.4 The one rule that needs new state: the tainted-path ledger

The brief asks for "writing a retrieved value into a path the model then reads"
to be covered, and `GUIDE.md:50-54` *sanctions* the redirect-to-file form
(contained, not a compromise — but cleanup debt, "delete the copy afterwards
without reading it"). Both statements are true, and together they leave the
programme's biggest single-call hole: an authorised redirect puts the real value
on disk, and a *later* call has no `lop secret` in it at all, so there is no
source pattern for the scan to key on.

Design: `execute_bash` records `(path, rule_label, call_id)` in a **session-scoped,
bounded, never-persisted** ledger when it approves a redirect (`> f`, `>> f`,
`tee f`, and the `lop secret file` case is *excluded* — that verb already removes
its copy, design §7). The read-side tools (`read`, `grep`, and `bash`'s `cat`)
then refuse to read a ledgered path and return the guide's instruction: delete it
unread. Bounded (an LRU, like `eval_worker`'s `_LEDGER_MAX_BYTES`,
`eval_worker.py:288-292`), cleared on `rm` and at session end, and *loud rather
than silent* when it fires.

Two honest caveats, because the ledger is the weakest part of this design: a path
reused for a legitimate purpose produces a false refusal (mitigated by clearing on
`rm` and by naming the file it refuses on), and a *different* session — a resume, a
sibling agent, a subagent with its own store view — has no ledger at all. The
minimal fallback, if the team prefers not to add cross-tool state now: keep the
redirect allowed and have the tool result carry a notice ("a plaintext copy now
exists at `<path>`; do not read it — delete it"). I recommend the ledger, because
the fallback relies on the model's compliance and the whole incident is what
happens when a model does not comply.

### 3.5 What the refusal returns

A refusal is a tool result, not an exception and not a prompt:

```
[secret sink] refusing to run: the value from `lop secret get GITHUB_TOKEN` is
consumed by `echo`, which prints it into this session's transcript.
Rewrite it as one of:
  curl -H "Authorization: Bearer $(lop secret get GITHUB_TOKEN)" https://…   (use it)
  lop secret run --secret GITHUB_TOKEN -- curl …                            (never crosses stdout)
  lop secret describe GITHUB_TOKEN                                          (identify it, no value)
rule: shell.print-of-source
```

Naming the rule label makes the refusal debuggable and makes the rule table
falsifiable in review ("this refusal came from rule X, here is rule X's `why`").

### 3.6 Unparseable text: fail closed only where it matters

The policy, and its justification against the true-positive cost:

* **no source present → never refuse.** The scanner's blast radius is exactly the
  set of commands that fetch a secret, which is the gating condition the brief
  asks for. A command that runs a build, a test suite, a `terraform apply` or a
  train loop cost nothing then and can never be refused for a lexing failure.
* **a source present and the surrounding text does not lex → `unresolved=True`,
  refuse, and say which span defeated it.** A false refusal costs the model one
  re-spelling (the rewrite hint in §3.5 carries the same information in a form the
  scanner accepts), which is recoverable and visible; a false allow is a
  credential read into the transcript, which is not recoverable at all. That
  asymmetry — not a claim that the scanner is accurate — is what justifies the
  fail-closed half.
* **a source present in a construct the scanner does not model at all** (a `sh`
  script invoked by name that a previous call wrote, `eval "$(…)"`, a
  base64-decoded command, `python -c` on the bash surface) is *not* an `unresolved`
  — the scanner cannot know to look — and it is precisely where A1's CLI default
  does the work. This is the reason §2.2's last column matters more than the rule
  table's completeness: **PR B's accuracy is a cost/latency question; PR A's
  default is the safety property.**

---

## 4. PR D: the classifier layer

**Recommendation: do NOT build PR D now.** Build the deterministic layer, measure
its residual, and revisit with data. The reasoning, clause by clause.

**The contract forbids the obvious home, and this is not a technicality.**
`docs/design/classification-layer.md` §6: *"The classifier's answers are
**advisory**. Nothing in this layer may gate a capability, change an approval
tier, or alter a tool's availability."* The module docstring repeats it as a
prohibition (`classification/__init__.py:20-21`, "Nothing here may gate a
capability, change an approval tier or alter a tool's availability (§6). A caller
that acts on a recommendation without checking the resource itself is using this
layer wrongly."). A guard that blocks a bash call is enforcement on a tool result
— and §6 also says the layer is *not* transcript-reading: "No history, no
compaction summary, **no tool results**" (`classification/__init__.py:27`).
Amending those clauses to host an enforcement path would be rewriting the layer's
purpose to serve a caller it was not designed for, and it would take the layer's
own fail-open shape ("Every failure mode returns an empty `Recommendation`",
§4/`__init__.py:28-30`) and invert it. If a model is ever in this path, it needs
its **own purpose-built seam with its own data-minimal contract**, not a
retrofit — that is the honest answer to the "new seam vs amendment" question.

**Why not now, on the merits:** the gating condition the brief proposes ("the
step contains `lop secret` usage") is *syntactic*, and the deterministic scanner
answers it without a model, offline, without latency, and — critically — must be
able to hold on its own anyway, because the specified fail mode is "if the
classifier is unavailable the deterministic layer must still hold (fail-closed to
A+B, never fail-open)". A classifier that can only ever *add* refusals on top of a
layer that must work without it is: latency on a hot path (`execute_bash`), a new
provider dependency on the security path, and a new data-egress surface — for a
class of decision (does this text's value flow reach a printing sink) where the
lexer is both cheaper and more precise.

**If it is built, these are the constraints it must satisfy, and they are not
negotiable:**

* **Deny-only.** The classifier may return `deny` or `undecided→deny`. It must
  never be able to authorise a reveal or a `supply`: a model that can grant a
  capability is a new attack surface on the control it is meant to strengthen.
* **Gating condition narrower than "contains `lop secret`".** The honest residual
  set is *the spans the lexer could not resolve* — i.e. text that already reached
  `unresolved=True`. Gating the whole class re-asks a question the lexer answered,
  and pays for it on every secret-bearing command.
* **Data-minimal, and the state itself is a hazard.** To judge a command the seam
  needs the command text, and a command text can contain the credential itself —
  which is why `harness/redaction.py:24-30` scrubs the argument summary before it
  is stored ("a credential can be typed into the call itself … and a report about
  a redaction must not be the next place the value appears"). So the seam's
  contract must carry a redacted form of the command, its own redaction boundary,
  a bounded size, and a stated "no history" rule — the §6 rails, re-derived rather
  than inherited.
* **Fail mode:** unavailable classifier ⇒ A+B unchanged and the call is refused
  (never allowed). That is the only fail mode consistent with §3.6's asymmetry.

**What would change this recommendation:** a measurement, not an argument. Count
`lop secret`-bearing commands the scanner leaves `unresolved` or refuses (a local
counter in the rule module, no egress). If a meaningful share of real usage lands
there, a model becomes worth its cost as a *second* refusal on exactly those spans
— and the first thing to try is still not a model: it is a better rule or a
better lexer.

---

## 5. PR C: where the holes actually are, and where the fix belongs

### 5.1 The chunk-boundary hole: exact location

`local_operator/tools/builtin.py::_PipeRedactor._release_point` (`builtin.py:2325-2363`),
called from `_feed_scrubbed` (`builtin.py:2243-2258`) which masks *after* the cut;
the retention and mirror callers are `_pump` at `builtin.py:2859-2920`
(`safe = redactor.feed(chunk)`, `sink.append(safe)`, `_mirror(safe)`).

*What is already right:* the lookbehind (`builtin.py:2207-2210`), the per-chunk
value refresh (`builtin.py:2211-2223`), and the "never cut through a KNOWN value"
loop (`builtin.py:2351-2362`).
*The defect:* that loop's `text.find(secret, …)` can only match a value that is
**entirely present in the accumulated text**, while the cut is placed after the
last newline in the chunk (`builtin.py:2333`) — so a value containing a newline is
released raw, prefix first, exactly as reproduced in §1.3(a). The same partial-value
blindness applies to the deferral cap (`builtin.py:2345-2350`), whose subsequent
hold-back loop is the same present-value check.
*Fix direction (no code here):* hold back the longest suffix of the accumulated
text that is a **proper prefix** of any known value, independently of whether the
whole value is present. That is a bounded computation (the bound is already the
longest known value, i.e. exactly `lookbehind`'s input) and it generalises the
existing idea instead of adding a second mechanism. Note the interaction to keep:
the cap must stay last so bounded memory still holds, and the prefix hold must run
after it.

*Exactly what the leaked bytes reach, stated narrowly because it is easy to
overclaim:* the settled tool result goes through `redact_tool_result` →
`VariableStore.redact_with_report` (`session.py:9314+`, `variables.py:525-556`),
which replaces whole strings, so a value the session has **registered** is masked
there. What is *not* covered: the live stream card, `jobs(op='peek')`'s buffer
(`builtin.py:2841-2856` `_mirror`), the abort receipt, and the **spill file's
bytes on disk** (`builtin.py:2905-2917`) — the last of which is a plaintext
credential in a readable file, i.e. exactly the containment failure the
`redaction_shapes` docstring says the harness treats as *not* compromised only
because it is not in the model's context. It is in the model's context again the
moment anything reads it without the same registration. **The evidence that would
settle the remaining question** (whether the settled pass covers every trigger,
including the retention-split case PR #1422 already recorded): drive `execute_bash`
with a real multi-line secret and read `spill://` — a test, not a design claim.
One reason to expect more residue: PR #1422's body records a *different* trigger
for the same outcome — retention dropping one PEM marker publishes ~1,958 body
lines raw, "end to end … identical in the spill file and served over `read
spill://`".

> **Amendment after shipping — measured on PR #1428's head (`f111faac`).** The
> fix direction above is *not* what shipped, and the defect is still live.
> Re-running probe A.2 against that head still reports `RAW-BYTE LEAK: True` with
> the same released bytes, and its `_release_point` shows why: the cut is still
> placed after the last newline, and the pull-back loop still searches for a form
> that is **fully present** in the accumulated text (`text.find(form, …)`) — so a
> multi-line value whose first line has arrived and whose second has not is
> invisible to it. #1428 did widen the hold, over every *spelling* of a value and
> over cap-forced cuts, behind a bounded window (`_STREAM_HOLD_LIMIT`), and it
> found a **second chunked seam this document missed**: the eval worker's
> streamed frames (`tools/eval_worker.py`, `_StreamingTextIO.write`), which
> masked per write and now release through a windowed masker at end of cell. The
> live case's scope, stated exactly: it needs a chunk boundary to fall inside a
> multi-line value — deterministic for a value larger than one
> `stream.read(65536)`, and for a line-buffered writer, but not for a small value
> arriving in one write (probe A.2's "whole" case masks correctly). A PEM body
> has its own independent masker on this path (`_mask_open_key_block`), so the
> exposed shapes are the multi-line values that are *not* PEM: a pretty-printed
> service-account JSON, a multi-line `.env`, any multi-line opaque value. The
> unshipped fix — hold back the longest suffix that is a *proper prefix* of a
> known value, independently of presence — is still the smallest one.

### 5.2 Transform normalisation: one home, and it is the shape module

The transform hole (§1.3(b)) is a property of the *exact-value* pass
(`variables.py:545` `scrub_secrets_with_hits(text, self.redaction_values())`,
`runtime.py:136-141` `_RedactionLedger.scrub`). The risk of fixing it where it
lives is obvious: a second copy of the mask policy, in a second module, drifting
from the first — which is the failure `redaction_shapes.py:18-27` says it exists
to prevent ("moved here so there is ONE table rather than a second copy per
surface").

**Recommendation:** put the transforms in `local_operator/redaction_shapes.py`,
exported as one small, pure API — `transformed_forms(value) -> tuple[str, ...]`
and a `mask_known_transforms(text, values) -> tuple[str, ShapeReport]`-shaped
caller — and have **both** consumers call it: `VariableStore.redact_with_report`
(the settled/live-text path) and `_PipeRedactor` (the stream path). Then the
policy is one module, the two surfaces cannot disagree, and the registration
back-channel (`variables.py:_register_shape_hits`) already exists for containing
what a transform match reveals.

Transforms worth having, in the module's own terms (context-anchored, all-or-none,
over-masking is a defect, `redaction_shapes.py:58-70`):

* **reversal** — the incident's actual bypass, cheap (reverse, replace, reverse
  back), and unambiguous;
* **base64 / base64url, hex, percent-encoding** — the plausible mechanical re-spellings;
* **NOT** per-character interleave or piecewise slicing. Probe A.3 shows they
  defeat the pipeline too, but covering them means n-gram matching, which is the
  "looks random"/loose-match direction the module explicitly refuses
  (`redaction_shapes.py:38-42`: "There is deliberately no 'looks random' /
  entropy rule"). Name them as residue in §6.3 instead.
* **A length floor** on every transform (a few bytes), because a tiny value's
  base64 or hex form occurs by chance in ordinary output, and the module's own
  doctrine is that over-masking is a defect, not headroom. The floor is a
  calibration decision for the implementer; the *rule* is that a transform match
  must be reported as an event (identity, not just mask) so the notice fires even
  when the mask succeeds.

> **As shipped (PR #1428), and where it differs from this recommendation.** The
> transforms live in the shape module, as recommended, with one home and both
> surfaces reading it. The shipped family list is **wider** than the one
> recommended here: reversed, base64 (standard and URL-safe, padded and
> unpadded), hex (`xxd -p`), percent (`quote`/`quote_plus`), JSON-escaped (raw
> and ASCII-escaped) and separator-spread (space, `-`, `.`, `:`, `\n`),
> enumerated for values of at least `_TRANSFORM_MIN_VALUE_LEN = 12` characters.
> The floor is the over-masking control that makes the wider list acceptable —
> precisely the mitigation recommended above for the interleave case this
> document had proposed leaving as residue, and the reason the list could be
> widened without breaking the module's all-or-none invariant. The PR's own
> stated residuals are the honest remainder: rot13, gzip, a double base64, a
> per-character shift, a value below the floor (verbatim only), and a spelling
> longer than its 64 KiB hold.

### 5.3 Coordination with the open work, checked rather than assumed

* **PR #1422** (`fix/redaction-step-cost`, open, `MERGEABLE`, 3 files:
  `local_operator/tools/builtin.py` +130/-20, `tests/unit/secrets/test_credential_shapes.py`
  +360, `tests/unit/tools/test_loop_liveness.py` +60/-7) touches the **settled**
  path and the decode/threading site (`_decode_and_redact_streams`,
  `_BashOutput.decode`, `_REDACT_STREAM_LIMIT_CHARS`) — *not* `_release_point`
  or `_PipeRedactor`'s masking logic. PR C therefore rebases on it and touches a
  different function in the same file; C must not touch
  `_BashOutput.decode`/`_redact_settled_stream`/`_REDACT_STREAM_LIMIT_CHARS`.
  Two further notes from its body worth carrying: it *corrects* the widely
  repeated "`hub resume` scrubs 22 transcripts" claim (`session/transcript.py`
  contains no redaction — the replay is a passthrough), and it records the
  retention-split PEM follow-up as its own change, which is adjacent to §5.1's
  fix and should not be duplicated.
* **Branch `f4ab88a4`** (`fix/bounded-credential-shape-scan`, "partial bound of
  the credential-shape scan (interrupted)") is based on `9405d6d6`, far behind
  `origin/main` at `0caf7a32`, and touches 7 files including
  `local_operator/redaction_shapes.py` +116 and `local_operator/tools/builtin.py`
  +104 — i.e. the **same functions** §5.2 wants to change (the shape pass and its
  callers). It is not a safe base: do not build C on it, and do not fix it as a
  side effect. Either it is revived as its own bounded PR (its owner's call) or
  it is dropped; C lands on main-after-#1422 and rebases if it revives.

---

## 6. Residual paths: does any path still put a value in the model's context without an explicit, audited, gate-able opt-in?

**Yes, and here they are.** These are the paths that remain after what actually
shipped (§0.2), ordered by how reachable they are from a *normal* turn. A named
hole is worth more than a clean-sounding summary, so nothing below is rounded off.

**Status tags**, applied per item and current as of the decision record:
*closed* — a shipped control now covers the path; *narrowed* — part of the path is
closed and the remainder is named; *open* — nothing shipped covers it; *guide-only*
— no control of this shape can cover it, so the only honest remedy is a change to
`guide://credentials` rather than a mechanism.

**6.1 The file on disk, read back later (the biggest one).** `lop secret get X >
/tmp/token` is sanctioned (`GUIDE.md:50-54`, and the brief requires the redirect
to be allowed). Under A1 the *harness* supplies those bytes, which means the
harness knows the path — hence §3.4's ledger. With the ledger: closed inside the
session, still open across sessions, resumes, sibling agents and subagents; and
the plaintext file itself is what anything outside the session reads.
*Without* the ledger: open by default, and the only net is the shape pass, which
is spelling-based (6.2). **Status: open — and now the largest one.** PR #1429
scoped §3.4's ledger **out** and recorded it as its own gap ("only a write *and*
read in the same call"), so today the redirect is allowed, the result tells the
model to delete the copy unread, and a later `cat` of that path hits only the
shape pass in-session — nothing at all across sessions, resumes and sibling
agents. **Partly guide-only:** the in-session, same-call case is a real control
waiting to be built; the rest is words in `GUIDE.md`, which #1429 and #1430 both
already touch.

**6.2 Values that never went through the seam.** A value the session never
retrieved is invisible to the exact-value pass by definition: the remote
environment (`kubectl exec … env`, the motivating case in
`redaction_shapes.py:7-20`), a value pasted by the user, one read from a log or a
file by a different tool. Only the shape pass applies, and its own docstring
states the residual: "It does NOT recognise an opaque value with none of these
spellings around it" (`redaction_shapes.py:35-40`). No opt-in exists to gate,
because the store was never asked. **Status: open, guide-only** — a control built
on "the store knows the value" cannot cover a value the store never saw, so the
remedy is instructional (do not paste, do not echo a remote environment, redact at
the point of use), not mechanical.

**6.3 Transforms outside the normaliser.** The transforms §5.2 shipped (see its
as-shipped note) cover reversed, base64, hex, percent, JSON-escaped and
separator-spread spellings for values of 12 characters or more. Outside them:
rot13, gzip/deflate, a double base64, a per-character shift, UTF-16 or latin-1
re-encoding, `od`/`xxd` variants the list does not enumerate, piecewise slicing
(`${v:0:8}${v:8}`, `v[:8] + v[8:]`), any custom re-spelling — and, for a value
under the floor, everything but the verbatim form. **Status: narrowed
(PR #1428); open for the list's own stated residuals.** A transform is a
*deliberate* act; the honest goal is that cheap transforms are *detected and
reported*, not that transforms are impossible.

**6.4 Live surfaces and the spill bytes (from §5.1).** Bytes the pipe filter
released raw (multi-line values) are in the live card, the peek buffer, the abort
receipt and the spill file. Masked at a *read* result only when the session has
the value registered; the spill's bytes on disk stay plaintext regardless.
**Status: narrowed (PR #1428) — and the multi-line case is still open.** #1428
replaced the eval worker's per-`write` masking with a bounded windowed masker and
closed the transform spellings on the pipe path, but it does not change the
release point's newline rule, so the multi-line straddle measured in §5.1's
amendment still releases raw bytes into the live card, the peek buffer and the
spill buffer. **Partly guide-only:** the fix is a mechanism (§5.1), but until it
lands, the instruction that matters — do not `cat` a file-shaped secret into a
live view — is a guide sentence.

**6.5 System-prompt blocks.** Named by the shapes module itself
(`redaction_shapes.py:43-50`): "the **system-prompt blocks**
(`prompts_api.build_system_blocks`: repo guidance, the skills block, the base
prompt). A credential sitting in a checked-in file that the session injects
therefore reaches the model verbatim; the session is not the source of that value
and cannot contain it." A credential in a checked-in file, a skill, a guide or
`AGENTS.md` reaches the model with no scrub, no notice and no opt-in. (The same
docstring names `mcp/redaction.py` as values-only — that one is *not* a hole,
because its surfaces are not model-visible.) **Status: open, guide-only.** No
control this programme builds is on that path: the session is not the source of
the value and cannot contain it. The only remedies are to keep credentials out of
checked-in files (a repository-hygiene rule, and the reason `guide://credentials`
says never to write one into the repo) and, for the operator, to treat any
credential that has ever sat in a tracked file as exposed.

**6.6 Surfaces the model reads that are only partly on the hook path.** The
loop's hook covers tool results (`loop.py:3090-3096`, `session.py:9314+`) and
scrubs tool-call arguments in stored history (`loop.py:1474`). What is thinner:
the per-line live text path (`builtin.py:2366-2385` `_bash_progress_line` →
`_redact_tool_text`), which cannot match a value spanning a line; and a
subagent's or peer's own transcript on disk, which is a separate session artifact.
I did **not** verify the scrub coverage of every non-tool channel (a `send`/`hub`
peer message, an `ask` question's text, a notice row) and I am not going to assert
it either way: the evidence that would settle it is a value-typed probe through
`ask` and `send` with a registered value, and that belongs in QA's matrix, not in
a design claim. **Status: narrowed / partly unverified.** #1429 refuses the
in-command printing shapes on these surfaces' inputs (a `lop secret get` in a
`send` argument is a source reaching a printing sink), and #1428's windowed
maskers cover the bash live path and the eval frames; the per-line text path and
the cross-channel question above are unchanged, and this document still declines
to claim either way.

**6.7 The default tier enforces nothing.** `access.py:360-388` plus design §8:
a broker *denial* falls through to a local decrypt in the `keyfile` tier, and the
tier is the security boundary. Everything this programme calls "gate-able" is
harness-enforced; the store's own boundary is only real after `lop secret harden`.
This is the reason §1.3(c) exists as a constraint rather than a footnote.
**Status: open by design** — this is a tier choice, not a defect, and the remedy
(`lop secret harden`) is the operator's.

**6.8 A TTY reached through a pty.** `--reveal` is refused without a TTY, and the
`console` tool drives a real TTY (`builtin.py:13750-13787`, whose description
explicitly invites the agent to *read* surfaces the user opened). So an agent can
reach a reveal by typing into a console surface. It is visible (the operator's own
console tab) and audited (`reveal`, outcome `tty`, with pid and executable), which
is the most this control can claim — it is an audited deliberate act, not a silent
one. **Status: narrowed (PR #1430).** The reveal is now `get NAME --reveal`, it is
refused unless **stdin and stdout are both a terminal**, an environment variable
is explicitly not an opt-in (measured: `LOP_REVEAL=1` changes nothing), and it
writes a `reveal` row distinguishable from the `get` a pipeline writes. The pty
route remains: an agent driving a console surface reaches a real terminal, which
is why the audit row — not the TTY check — is the control that matters.

**6.9 `ps -ww` and friends inside an approved call.** With A1 the value lands in
the consumer's argv or environment; any same-uid reader in the same call can lift
source, the *general* case —
a value in argv/env while another process is watching the machine — is the
store's long-standing §2.2/§9 residual. **Status: narrowed (PR #1429)** for the
in-command shapes it can see; **open** for a watcher outside the command (another
process sampling `ps` or `/proc` while the value is in argv), which no
pre-execution scan can close.

**6.10 The model already holding the value.** Once a value is in context (by any
of the above, or by the user pasting it), nothing in this programme gates its
re-use, re-spelling or retyping into a response, a peer message, a scratchpad, a
commit message. The guide forbids it (`GUIDE.md:151-166`); no mechanism enforces
it. This is the ceiling of a control built on top of "the model does not have the
value". **Status: open, guide-only** — and it is the reason the programme's other
items matter: every closed path above is one fewer way to *get* here.

**6.11 The deferred `get` default: raw bytes for callers the scan cannot see.**
Because R1 is deferred (§0.1), `lop secret get` still writes exact bytes to stdout
for **every** caller. What stands in front of that is #1429's scan, which refuses
the shapes it can see and — by its own recorded design — does not model a script
invoked by name (`bash scripts/deploy.sh`, whose body interpolates `lop secret
get`), a `python -c` on the bash surface, an encoded or generated command, or flow
across separate calls. So the incident's *form* still works from those routes:
the value is fetched raw, and the nets behind it are the exact-value pass (only
for a session that registered the value) and the shape pass (only for spellings
it knows). **Status: open, and it is the price of the deferral** — stated so the
deferral is a decision rather than an oversight. The remedy is the R1 follow-up in
§0.1, which needs the operator's word; until then the honest position is the one
`GUIDE.md` already takes, that the harness refuses what it can prove and the rest
is the operator's residual risk. **Guide-only in the meantime.**

**Direct answer to the operator's question.** *Yes.* After everything that
shipped (B, C, and the additive half of A — §0.2), a value can still reach the
model's context with no explicit, audited, gate-able opt-in, through:

* **6.11** — `get` still prints raw bytes for a caller the scan cannot see (the
  consequence of deferring R1). The largest one, and the only one a *decision*
  rather than a bug fixes;
* **6.1** — a sanctioned plaintext copy read back later: closed in-session by the
  ledger §3.4 (not shipped), open across sessions, resumes and sibling agents;
* **6.2** — a value the store never saw;
* **6.3** — a transform outside the shipped list, or any transform of a value
  below the 12-character floor;
* **6.4** — bytes the pipe filter released raw (the multi-line straddle §5.1's
  amendment measures as still live);
* **6.5** — a credential in a checked-in file the session injects into its prompt;
* **6.6 / 6.7 / 6.9** — the surfaces and tiers whose coverage is partial or by
  design;
* **6.10** — re-use of a value already in context.

Which of them only a guide change can address: **6.2, 6.5, 6.10, and 6.11's
interim state** — no control of this programme's shape reaches them (the store was
never asked; the file is not the session's; the value is already in context; the
caller is outside the harness). Those belong in `GUIDE.md`'s "What this actually
protects against" section, stated plainly, rather than left for the programme's
success to imply otherwise. **6.1, 6.3 and 6.4 are mechanical and remain worth
building** (the ledger, the remaining transforms, the proper-prefix hold), and
**6.11** is the operator's call per §0.1.

---

## 7. PR split, sequencing, and what each PR's evidence must show

Ordering matters: A's default is what makes B's misses survivable, and C is the
accident net underneath both.

| PR | scope | does not include | evidence it must carry |
|---|---|---|---|
| **A** | `get` = value-or-nothing (exit 3 + actionable refusal); `--reveal` TTY gate; `describe` gains length + keyed fingerprint; audit events of §2.3(5); guide re-pointing to `run`/`file` | the scan; the transforms | reproduce the incident command on the current build (raw values in the result), then the same command refused with the rewrite text; a real `curl` against a local endpoint through `run`, `file` and `$()` showing identical bytes and identical requests; the audit rows for each path; a non-TTY reveal refused |
| **B** | `harness/secret_sinks.py` (lexer + `RULES`), the refusal in `execute_bash` and `execute_eval`, the `supply` verdict A1 consumes, the tainted-path ledger | the transforms; anything in the classifier layer | the four required non-findings as counterexample tests *plus* live spells of each; a refusal per sink class with its rule label; an unparseable case refused with the span named; the ledger's refusal of a read of the redirected path; the eval cell equivalent |
| **C** | the proper-prefix hold-back in `_release_point`; the transform set in `redaction_shapes.py` consumed by both surfaces | PR #1422's territory (see §5.3) | a multi-line value split at a newline, before/after, through `_PipeRedactor` *and* through `execute_bash` into the spill; the transform corpus with the negative set intact |
| **D** | nothing yet — measure, then decide | — | a counter of `unresolved`/refused secret-bearing commands, reported as a number |

**Not in scope, deliberately:** a second approval prompt anywhere (A1 needs no
approver for the sanctioned form); any change to the broker's protocol; a
`--values` flag on `list` (design §5.1 refuses it for a reason that has not
changed); a version bump on any of these branches.

---

## 8. Risks to watch during rollout

1. **The sanctioned form must not break in the field.** The single highest-cost
   failure of A1 is a rewrite that changes semantics (quoting, word-splitting,
   `IFS`, a value that legitimately contains a newline). Mitigation: a
   byte-for-byte equivalence test per quoting context, and — since the value is
   registered either way — a `cmp`-style probe through a real consumer.
2. **Silent functional breaks.** Under A's default, a caller whose sink the scan
   could not prove gets an exit-3 refusal *inside a command that may not check
   its status*. The refusal text must reach the model (it does: tool output), and
   the guide must name the two-step recovery. Watch for a spike in exit-3s during
   the first days — that is the measurement that sizes PR D (§4).
3. **False refusals on heredocs (B).** The highest false-positive class, and the
   one that will annoy real work fastest. Corpus-first, then ship; keep the rule
   that a refusal always names its rule label, so a false positive is
   attributable to one rule and one test case to add.
4. **Over-masking from the transform normaliser (C).** The module's own doctrine
   (a mask is all of the credential or none of it; over-masking is a defect) means
   the transform set needs its own negative corpus, especially for short values.
5. **The tainted-path ledger's false positives** (§3.4) and its cross-session
   gap, which must not be described as coverage it does not have.
6. **The broker-side temptation.** Anyone implementing A will be tempted to put
   the gate in the broker because that is where §6's ack already lives. §1.3(c)
   says why that is not a gate in the default tier.
7. **The notice must keep telling the truth.** A refusal is not a redaction, and
   `incidents.format_shape_incident_message` (`incidents.py:1008-1044`) has two
   carefully-different texts; a third ("refused before it ran") must not be
   worded as either, or an operator will rotate a credential that was never read,
   or not rotate one that was.

---

## 9. Verified, assumed, and open

**Verified by reading the code (file:line above):** the `get` stdout contract and
its routing through `retrieve_secret`; the absence of any pre-execution guard;
the ledger and its longest-first `replace`; `_list`/`SecretRecord`/`secret_tool`'s
value-free shapes; the audit chain's value-free rows and free-text `event`; the
`BrokerDenied` → local-decrypt fallback in the default tier; `describe`'s
decrypting behaviour and the `_payload` layout; the loop's scrub call sites; the
shape module's own statement of the two surfaces it does not reach; the
classification layer's advisory/not-transcript-reading contract.

**Verified by running it:** the exact bytes and missing trailing newline, the
`deny:retrieve` → `get ok` fallback in a real isolated default-tier store, the
single-line straddle being *handled*, the multi-line straddle leaking, and the
transform table in §1.3(b). Probe scripts and raw output: Appendix A.

**Assumed, not verified:** that the settled result path masks a multi-line value
in every trigger combination (§5.1 states what would settle it); the scrub
coverage of non-tool channels — peer messages, `ask` question text, notices
(§6.6); that no *other* surface produces `lop secret get` bytes outside
`handlers._get` (the argument is structural — one writer of the value bytes —
but I did not exhaustively grep for a second writer).

**Amended after shipping.** §0 records what was implemented, and §5.1's amendment
and §6's status tags were added from re-measurement against the shipped heads
rather than from their summaries: §5.1's multi-line case was re-run on PR #1428's
head and still leaks, and the eval worker's streamed frames were a second chunked
seam that this document did not find on its own (credited there).

**Open questions the implementer should settle by test, not by argument:** the
exact set of quoting contexts the A1 rewrite must reproduce byte-for-byte; whether
the `supply` verdict should be refused (rather than rewritten) when the value
would land in an environment variable that the child might dump
(`env`, `set`, `printenv` in the same command are sinks and belong in the table);
and the transform length floor for C.

---

## Appendix A — probes and raw output

All probes are read-only and import the real modules from the worktree at
`origin/main` (verified by printing `local_operator.tools.builtin.__file__`).
Environment: `env -i HOME=<fresh iso> LOCAL_OPERATOR_CONFIG_DIR=<iso>/.local-operator
PATH=… TERM=xterm-256color`, `PYTHONPATH=<worktree>`, the repo's own
`.venv/bin/python` (3.12.13). Synthetic values only.

### A.1 The stdout contract and the tier fallback, against a real isolated store

```
$ printf %s 'probe-not-a-credential-5f3a' | lop secret set PROBE_KEY
stored PROBE_KEY (27 bytes, kind=string)
$ lop secret get PROBE_KEY | xxd | tail -3
00000000: 7072 6f62 652d 6e6f 742d 612d 6372 6564  probe-not-a-cred
00000010: 656e 7469 616c 2d35 6633 61              ential-5f3a
$ lop secret get PROBE_KEY | wc -c
      27
$ lop secret audit --limit 5
2026-09-22 09:40:47  get     ok    pid=98996   1cabbd78-990b-48e2-9c1d-6c95d2cbee53
2026-09-22 09:40:48  deny:retrieve deny  pid=99044
2026-09-22 09:40:48  deny:key deny  pid=99044
2026-09-22 09:40:48  get     ok    pid=99044   1cabbd78-990b-48e2-9c1d-6c95d2cbee53
2026-09-22 09:40:49  deny:key deny  pid=99091
```

27 bytes exactly, no `0a` — and the `deny:retrieve` line followed by `get ok` for
the same pid is §1.3(c): the denial was a fallback, not a refusal.

### A.2 `_PipeRedactor` and the chunk boundary

Measured against `origin/main` at `0caf7a32` (the base this document was written
against), worktree `secret-get-hardening`, with synthetic values chosen so this
record carries nothing credential-shaped — an earlier run used an issuer-prefixed
token and the harness's own shape pass flagged it on every read, which is why the
values here are neutral. Two echoed chunk reprs in the captured output were masked
by the session's own redactor while it was captured; they are shown as they came
back.

```
module under test: <worktree>/local_operator/tools/builtin.py

[single-line split] value is 28 bytes
  chunks=[b'prefix probe-value-01', b'23456789abcdef suffix\n']
  released=b'prefix [redacted] suffix\n'
  RAW-BYTE LEAK: False

[single-line whole] value is 28 bytes
  chunks=[b'out: [redacted]\n']
  released=b'out: [redacted]\n'
  RAW-BYTE LEAK: False

[multi-line split at newline] value is 29 bytes
  chunks=[b'out: probe-line-one\n', b'probe-line-two\n']
  released=b'out: probe-line-one\nprobe-line-two\n'
  RAW-BYTE LEAK: True

[multi-line whole] value is 29 bytes
  chunks=[b'out: [redacted]\n']
  released=b'out: [redacted]\n'
  RAW-BYTE LEAK: False

[single-line, value then newline] value is 28 bytes
  chunks=[[redacted], b'\n']
  [redacted]\n'
  RAW-BYTE LEAK: False
```

The leak needs the value to contain a newline *and* a chunk boundary to fall
inside it: every chunk holding a newline that is inside a multi-line value
releases the part before that newline, and by `final=True` the remainder sits in a
buffer that no longer holds the whole value, so nothing matches. Re-measured on PR
#1428's head (`f111faac`, checked out into a throwaway worktree and removed
afterwards): **`RAW-BYTE LEAK: True`, same released bytes** — see §5.1's
amendment.
### A.3 Transforms: what the pipeline masks, and what only the shape pass catches

Value: `opaqueToken9f3a1c77d4e5` (a value the session knows).

```
verbatim, bare          exact+shape -> 'here it is: [redacted]'          hits=0
                        shape only  -> 'here it is: opaqueToken9f3a1c77d4e5'
verbatim, assignment    exact+shape -> 'TOKEN=[redacted]'                hits=0
reversed, bare          exact+shape -> 'here it is: 5e4d77c1a3f9nekoTeuqapo'   hits=0
reversed, assignment    exact+shape -> 'TOKEN=[redacted]'                hits=1
base64, bare            exact+shape -> 'here it is: b3BhcXVlVG9rZW45…'   hits=0
base64, assignment      exact+shape -> 'TOKEN=[redacted]'                hits=1
hex, bare               exact+shape -> 'here it is: 6f7061717565546f6b…' hits=0
per-char interleave     exact+shape -> 'here it is: oXpXaXqXuXeXTXoXkXe…' hits=0
piecewise, bare         exact+shape -> 'here it is: opaqueTo ... ken9f3a…' hits=0
```

Read it as: the exact-value pass covers verbatim; the shape pass covers a
transformed value only where it is spelled after an assignment; and reversal,
base64, hex, interleave and piecewise all pass *both* passes in a bare position.
