# Future-live owner-runtime file provenance (P2)

This is metadata groundwork for **Allow files the transcript names** with
**Use local device approval**. A provenance proof is **not a file grant**. P2
adds no outside-scratchpad read/list/bytes route, cache, Library entry, UI switch,
public proof issuer, audit service, identity registry, or device enrollment.
Existing scratchpad/attachment authority and the P1 approval domain are unchanged.
The S2/S3/P3 serving integration is deliberately absent.

## Original native event, not a transcript scan

An owning Session freezes metadata on a future-live admission or native executor
boundary. User `owner_typed` remains the existing locality/deletion-protection
flag, **not eligibility**. A local input and an authorized remote DRIVE/phone
prompt or steer both qualify as an owner-runtime admission; neither asserts a
verified human, account, native presence or P1 association.

The runtime derives its private user-admission context from the real checked
`_ConnectionAuthority` and the actual prompt/steer capability, not frame/body
labels. Serving freezes before its earliest queue/spool step. The process inbox
and first-turn Session inbox explicitly request continuation: only an original
same-runtime handle is usable. A cold spool after runtime loss stays unknown,
including an owner-looking `SOURCE_USER` row. It is never statted at drain time.
No imported/replicated/seeded history, peer/child/stdout JSON, role string or
unsealed dictionary can recreate the nonserializable capture ticket.

Other production boundaries are:

* the finalized assistant before its message-end event is yielded;
* an actually executed tool's original arguments and immediate completion,
  before post hooks/result copies, then binding to its newly allocated **result
  Message ID**, not just the tool-call ID;
* nested eval's awaited host-created original-call and completion bookkeeping
  events, never a cell-returned lookalike;
* native non-child async-job registration carrying its original private executor
  context, followed by the **first** coroutine completion before roster update,
  delivery, wait or peek. Persisted roster data cannot recover that context;
* the actual local bang executor immediately after `execute_bash`, before receipt
  queue/render work. `record_shell` alone is also an adoption/import sink and
  cannot issue. An attached/remote receipt has no owner-local native hook.

Ordinary tool/web/peer text does not count as owner adoption. The assistant can
later explicitly adopt a path in its own retained text. Typed write/edit/browser
path results are native output facts; arbitrary output text is not. Generated
media represented only by a content-addressed attachment digest does not invent
an outside-PAD path: its existing attachment authority stays separate. Unknown
or unavailable producers never become a delayed filesystem inventory.

## Frozen file and owner context

Capture records only path/reference metadata and file identity. It opens through
the shared S1 no-follow/original-component-safe primitives, requires a regular,
single-link file, applies the existing size bound and sensitive/relocated-config
refusals, and closes the descriptor immediately. No target content is read,
hashed or copied, and no file descriptor survives a turn. Raw `..` is refused
rather than normalized away; globs, unknown expansions and ambiguous cwd/code
remain unproven. No directory sweep is used.

The pure reference, markdown, explicit quoted literal, structured-argument and
shell/Python leaves are shared with the existing parser module. The new shell
input leaf handles conservative literal file operands using the existing token,
heredoc, expansion and lexical cwd helpers; it does not change inventory's
existing output/sweep policy. The Python leaf uses AST file-argument positions,
so `Path('out').write_text('contents')` does not attest a file named `contents`.
Computed values, comments/strings containing code and shadowed constructors do
not supply literal file operands.

A ticket also freezes the existing owner public key, physical config-root
identity, session identity, known birth and any already initialized proof
incarnation. Commit cannot silently rebind an old ticket after owner-key/root/
incarnation changes. First-ever materialization may bind the deferred session
birth and acquire the existing writer exclusivity **without restatting targets**.
Missing owner context or unknown writer exclusivity means unknown provenance,
not a new identity or a changed ordinary-transcript lock policy.

## Ordered durable publication

The only positive path is:

1. A sealed native ticket holds frozen metadata; no producer action is a grant.
2. Under the transcript's existing ordered writer lock, serialize the actual
   retained source bytes. A redacted-away or mismatched reference cannot bind
   from private original arguments. Native outputs additionally bind their real
   call identity and exact originating call-event bytes.
3. Write strict owner-private **pending** point records. Ordinary source events
   and a content-free opaque-ID/digest receipt enter the **same fsynced batch**.
4. Before the cancellation-shielded worker acknowledges that batch, publish each
   strict **committed** point record with exact source/receipt/origin byte spans
   and digests. A proof-finalization failure does not undo ordinary conversation
   persistence, and it does not cause target reinspection/retry issuance.

The private per-session `.file-provenance` directory and initialization sentinel
use the existing strict opt-in writer, parent/new-directory fsync and readback.
P1's old writer/defaults are not changed. Partial initialization does not repair
itself from history. A repeated ticket does not publish a second proof. Private
records and the sentinel are excluded from the real fork and network sync copy
inventories. Transcript exports carry only ordinary source text and content-free
bookkeeping, never the private records or file-identity/path metadata.

Native source facts and receipts are `ENTRY_CUSTOM` **bookkeeping**, not chat
messages. The private append path classifies them even for empty/failed captures.
This is independent of `details.hidden`: existing context/audit/legacy decoders
and the actual display-window DTO already skip bookkeeping. No new client must
learn a hiding flag; no phantom conversation row or new interaction is intended.
Default/no-ticket transcript row serialization remains unchanged.

## Point verification, failure and limits

A separate consumer process can call the internal opaque-ID point reader after
issuer exit or restart. It reads only the relevant private point record and exact
transcript byte spans, verifies owner/session/root/birth/incarnation, closed state,
record/source/receipt/origin digests and call pairing, then compares the current
file's device/inode/ctime/mtime/size/link/root identity with the frozen tuple.
It never searches history to rebuild a positive or promotes pending state on
restart. There is no issuer weak reference, PID affinity, all-history authority
map or new SQLite framework. Initialization reuses the writer's existing
constant-time latest-custom index to recognize surviving initialization knowledge.

Missing/corrupt/partial/private-mode/redirected state, source mismatch, changed
identity, lost capture, failed proof publication or unsupported durability means
**unknown**. Parsing/provenance failures must not fail an ordinary prompt, tool
result or shell receipt. Fully persisted committed state whose response was lost
can still verify; a pending-only crash cannot. File mutation never refreshes an
old proof. The records live and are removed with their owning session's normal
retention cleanup. Storage is proportional to issued point records; capture work
is proportional to the current literal candidates, not historical transcript
length. This slice does not claim a measured whole-runtime RSS bound or introduce
an unmeasured new timeout/size policy.

Owner-private state integrity remains a trust boundary. This is not a hostile
owner/OS anti-rollback service. The corrected P1 boundary in
[file-device-approval.md](file-device-approval.md) is unchanged: authoritative
negative witnesses are trusted; coherently rewinding **all surviving transition
knowledge** is unsupported and may require restoring only the journal after a
failed projection, not every physical file.

## Held integration dependencies

Linux file-only/staged native approval remains **denied**, exactly as P1 left
it: this is a coverage gap, not a fixed path. No credential borrowing, phone
fallback or new per-action ceremony is added here. Closing it is follow-on
**(B)**, owned by the guardrails lane and scheduled after #4098: the operator's
Mac/phone answers a question raised on the node by its `question_id`, bound
through the #4030 `bridge_decide` operator-binding anchor. P3 Linux rollout
waits on (B). P2 contains no code for (B) and does not sign it off. S2/S3 remain held until
P3; P3 still requires its mesh-owner review. The dormant P1 write-unconfirmed
retry metadata/copy correction is also deferred **before P3/UI**; P2 does not
claim to fix it. Prior relay-comment nits remain for the next relay delta.

## Validation boundaries

The focused tests use actual owning Session/harness/Serving/transcript worker,
real isolated file metadata, exact JSON and receipt readback, native nested/job/
shell callbacks, adversarial retention/failure controls and separate W/R process
lifetimes. Providers, mesh admission facts and host tool implementations are
synthetic, not canonical-account/native-presence/hardware proof. No live user
config, credentials, device peers or outside-PAD serving is used. Independent
security/code review and real-path QA on the pushed head are separate gates;
local unit counts are not a substitute. No browser, native UI or rendered-frame
claim accompanies this metadata-only slice.
