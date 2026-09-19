# Desktop control surface

This extends [DESKTOP_API.md](DESKTOP_API.md). It is a backend/transport contract,
not a claim that the desktop renderer or release is complete. The renderer still
must implement the destinations, authenticated stream relay, notification delivery,
reconciliation and visual acceptance. No version, package, PR or release is changed.

All routes below require the existing process-lifetime desktop bearer and exact
Origin policy and return no-store CRUD envelopes. A missing token is 401, an
unapproved Origin 403, and an unconfigured desktop capability 503. Validation
errors omit input values. Do not put credentials in URLs, slash arguments, logs,
renderer persistence or command receipts.

## Catalogue and dispatch

`GET /v1/desktop/commands` returns the shared `slash_commands.SLASH_COMMANDS`:
name, description, aliases, ArgumentMode, echo, consumes_prompt, prefixes_text,
argument_shape, argument_words, destination and execution host. There is no
desktop copy of the command list. The registry carries its destination; the
dispatcher only selects execution handlers.

The three text fields answer ONE question between them — is the text typed after
this command's word an ARGUMENT the command owns — which is what the messages
endpoint's admission rule and the composer's planner both read:

* `consumes_prompt` — free text destined for a model (`/goal <objective>`);
* `prefixes_text` — a value the composer can COMPLETE from a list (`/model gpt-5`);
* `argument_shape` / `argument_words` — the third source, for text the desktop
  VALIDATES or FORWARDS rather than completes. `argument_words` is the vocabulary
  the first token must come from, empty meaning any word:
  - `word` — one whitespace-free selector token (`/usage on`, `/stop now`);
  - `provider` — one token naming a provider this install knows, so `/login
    openai` is the command and `/login zzz` is a message;
  - `subcommand` — `<subcommand> [name]`, the MCP shape, at most two tokens, so
    `/mcp logout` is the command and `/mcp logout seems to cause a crash` is a
    message;
  - `any` — the command owns its trailing text, whatever it says (`/rename
    <title>`, `/move <path>`, and every row the two booleans above already
    carry);
  - `none` — NO source at all: the booleans above are false for this row and no
    shape applies, so text after the word is a message (`/compact hello`, which
    the desktop runs and silently discards).

  A third kind of row publishes `any` for a text NOTHING consumes: the command
  route REFUSES it, so the text is neither prose nor a handler's input
  (`/credential <key> <value>`, answered with "Enter credentials in the masked
  credential form, not command text"). `none` there is what let a whole-draft
  `/credential <secret>` be admitted as a MESSAGE on `/messages` while the route
  refused the identical text.

  **Precedence**: `consumes_prompt` and `prefixes_text` decide FIRST and are the
  whole answer when either is true; the shape is asked after them for text
  neither can describe. A row the booleans carry publishes `any` rather than
  `none`, so reading the shape alone, or OR-ing all three, gives the same answer
  — `none` never means "ask the booleans". The field is emitted on every row
  (never omitted), so no consumer has to infer a default.

  **A row that offers a value list does not publish `none`.** `ArgumentMode`
  (`arguments` on the wire) says a space here opens a list, so the command takes
  text and the shape must name where that text goes. The two fields disagreeing —
  `optional` beside `none` — is what admitted a whole-draft `/credential
  <secret>` as a MESSAGE while the command route refused the identical text,
  which is a raw credential on a paid turn for any client that plans prose from
  this field. Pinned for every row, so a future entry cannot reintroduce it.

A renderer applying those fields reaches the endpoint's own decision; one that
does not read them keeps its own derivation instead (both keys are additive).
The two questions differ on purpose for a SELECTOR: `/usage more prose` is prose
for the admission rule, while the command route still forwards it as a panel
selection.

`POST /v1/desktop/sessions/{id}/commands` accepts the existing stable UUID
request_id, command, args and optional images. Every canonical name and alias is
accepted. Runtime actions return the real SlashResult. Interactive/native actions
return `kind: native_action`, `destination`, `session_id`, `args`, typed `fields`
and destination-specific `data` (sources, submit paths, scope and safety flags).
This is a **presentation request**, not a receipt that the window closed, clipboard
changed or a destructive operation happened. Native receipts are retryable under
the existing at-most-once receipt protocol; the renderer must consume one action
once per command receipt, not once per reconnect.

`GET .../{id}/command-entities?command=...&name=...` supplies current models,
model-specific effort choices, approval choices, teams and attachable profiles.
Team `name` resolves through `org_chart.resolve_org`, including nested teams and
unresolved nodes. Agent choices use the shared role/specialist/packaged-seed
resolver; ordinary chat agents are not silently offered as personas. Details for a
selected profile include its actual instructions/tool restrictions. `/team chart`
is navigation, not an attempt to attach a team named `chart`.

Defaults use the existing typed settings API; session model, effort and approval
mutations do not silently persist. `/approvals default ...` opens the default
editor for `tool_approval_mode` and explicitly leaves the current session alone:
that editor writes the file, which is where every NEW session reads the mode, so
a running session's gate is loosened only by `/approvals auto` in that session
(route: `POST /v1/desktop/sessions/{id}/commands`). The file still tightens every
running session at once, which is the safe direction.
The frontend must obtain explicit default scope and premium-pricing consent in
its forms. Retain locally selected images while presenting an interactive action.

### All-command acceptance table

The backend census test invokes every row and every alias. “Native” below means
an actionable destination exists, **not** that its renderer acceptance is done.
Every UI responsibility in the last column still needs frontend implementation
and rendered verification.

| Command / aliases | Backend behavior | Frontend responsibility |
|---|---|---|
| help | Registry-backed `commands` destination | Searchable palette, aliases/arguments |
| exit / quit | `window.close`, unsaved guard, detach-only | Native close without stopping the runtime |
| clear | `transcript.clear`, view-only/history untouched | Clear painted rows only |
| copy | `transcript.copy`, history source and message/code/quote choices | Keyboard picker and clipboard |
| new | `sessions.new`, canonical creation endpoint, cwd field | New conversation, preserve current work |
| reload | `sessions.reload`, same canonical identity | Reopen/relaunch using installed runtime without resubmitting a turn |
| update | `updates`, capabilities source and retained identity | Existing updater/compatibility UI |
| resume / recall | `sessions.resume`, canonical list/source | Cold/live session picker and attach |
| rename | Runtime rename; bare name editor | Form and explicit-name receipt |
| fork | `session.fork`, canonical next-safe-boundary fork endpoint | Boundary explanation, child navigation, optional request once |
| model / models | ProviderController catalogue and runtime model change | Search/filter model picker, explicit default scope |
| effort | Model-specific entities and runtime effort change | Current choice/picker, unsupported model explanation |
| fast | Runtime control; native form marks premium pricing | Pricing warning and on/off control |
| theme / themes | `appearance`, desktop scope | Existing twelve-theme palette; terminal theme separate |
| provider | `providers`, central provider/auth sources | Provider grid, real states and method choices |
| settings / config | `settings`, full existing registry | Typed searchable editors, scope/reset |
| search | `settings.search`, web-search filter | Settings filtering and masked key entry |
| accounts | Central redacted accounts + account-specific removal | Account selection/confirmation and environment fallback |
| failovers | Selected/effective runtime models + configured default chains | Distinguish actual serving model from selected/default route |
| usage | ProviderController cached/live normalized reports and age/state | Freshness, quotas, partial/error/re-auth views |
| context | Runtime typed context result | Unknown vs estimate, breakdown display |
| analytics | AnalyticsStore aggregate/daily queries | Query controls and cost-knowledge rendering |
| goal | Runtime show/set/clear | Goal form; a set goal also consumes the request once (`goal_set`, same admission rule as team/agent) |
| loop | Runtime-local cancelable count/goal orchestration, snapshot state | Form, progress/judge state, cancel; never auto-answer gates |
| btw | Runtime completion, off-record panels, explicit adoption | Aside panel and adoption confirmation |
| compact | Existing runtime compact control/events | Pending/completed/error from canonical events |
| stop | Explicit target list/confirmation, canonical stop protocol | Current/selected/all picker; submit exact IDs |
| approvals | Runtime mode; explicit default editor writes the file, which loosens no running session but tightens every one | Session/default scope and confirmation |
| skills | Effective discovered catalogue and closed skill:// detail resolver | Catalogue/details; distinguish discoverable from selected |
| mcp | Effective source ownership, configuration, connections and grants | Server panel, forms, transport/downstream auth distinction |
| login | Central provider/method action and existing auth operation | Browser/input/cancel flow without renderer secrets |
| logout | Central provider/account selection and removal | Explicit confirmation; no implied environment removal |
| credential / cred | Masked form, runtime VariableStore operations | Secret input only, never composer echo/history |
| team / teams | Team registry/chart and runtime attachment | Team/request form; admission already consumes request once |
| agent / agents | Shared persona resolver and runtime attachment | Profile/request form; admission already consumes request once |

## Lifecycle endpoints

All paths start `/v1/desktop/sessions/{id}` unless noted.

- `POST /working-directory`: request_id and cwd, the live-session half of the
  terminal's `/move` and the same implementation. `cwd` resolves against the
  SESSION's current directory (`~` and `../sibling` accepted); the answer is
  `{cwd,label,outcome,will_wait}` with outcome `cold|rebound|unchanged`. The
  new directory is durable in BOTH `desktop.json` and the bridge's own `cwd`
  before anything is retired, and a refused move restores both. THE SUCCESSOR IS
  ENGAGED BY THE RETIRE FRAME, NOT BY THIS REQUEST: the runtime leaves by the
  `retiring` route, the viewer goes cold and the bridge re-engages eagerly, and
  the successor's bind is what republishes `frontend.cwd`.

  Two failure shapes, and the difference matters to the client. An ordinary
  refusal is a `409` carrying the session's own sentence (mid-turn, runtime too
  old to be moved, work that arrived during the retire, absent/unenterable path,
  unreadable/unwritable marker, another client attached, an older desktop
  window). An UNKNOWN owner outcome — the retire request left the process and no
  definitive answer came back, or a rollback could not run — is a `503
  {"detail":{"code":"move_outcome_unknown","message":...}}` (`code` and `message`
  ride under `detail`, the envelope every other error body on this API uses and
the one the desktop client reads): nothing is rolled back, the
  receipt stays pending, and the client reconciles against actual owner state
  before claiming either directory. The request is also at-most-once
  (`retry_safe=False`): a pending row is answered 409 indeterminate rather than
  re-executed, and a client that still wants the move mints a NEW id.

  EXCLUSIVITY IS BOUNDED, ON PURPOSE. A move retires the runtime and every
  attached facade then engages a successor from its OWN cwd, so the route refuses
  while another actual attach is registered (`409 … open in another terminal or
  attached client. Disconnect that client, then move again.`) and while a mounted
  desktop viewer has not negotiated the replacement frame (`409 … Update the
  desktop app, then move again.`). Several desktop windows behind ONE bridge are
  one attach and are unaffected. A TUI-initiated `/move` keeps the legacy,
  non-exclusive shape — mixed-viewer propagation is a non-goal of this release.

  Gated by `features.session_move >= 2` AND `features.frontend_replace >= 1`
  (the accepted directory reaches an already-mounted viewer through the
  `frontend.replace` frame, not through a same-sequence delta); `move` stays out
  of `OWNER_COMMANDS`, so a bare `/move` remains a native_action that asks the
  renderer to open its picker. Subagent children are separate sessions on their
  own leases and are NOT moved with their parent — a move mid-subagent leaves the
  child in the old tree.
- `POST /credentials`: action `list|store|forget`, optional key, secret value only
  for store, confirmed=true for forget. It calls the runtime's `credential_op`. Values
  never enter the command receipt database or transcript; only key names are
  journalled by the existing runtime. `/credential <anything>` is rejected rather
  than accidentally recording a secret — on BOTH routes, and for the same reason:
  `/commands` answers it with "Enter credentials in the masked credential form,
  not command text", and `/messages` refuses the whole-draft form as a command
  because the catalogue publishes `argument_shape: any` for the row. So the text
  reaches the masked form or nothing; it is never a message. Names-only listing
  does not expose values.
- `POST /mcp/credentials`: the MCP-only encrypted write, `{name, values:
  Record<secretId, SecretStr>, confirmed_replace: string[]}`. It is deliberately a
  SEPARATE route from `POST /credentials` above, which is the provider/session
  credential surface: this one stores into the encrypted secret store and never
  touches `credentials.env`, never enters the `/credential` variable store, and
  therefore never joins the environment of every unrelated `bash` child. Every
  submitted ID is validated against the named server's own declared `${NAME}`
  references BEFORE any write, so an unknown server, a config FIELD name
  (`Authorization`), an undeclared ID, an extra field, an empty value or a
  missing replacement confirmation returns a coded refusal
  (`{name, saved_ids, failed_ids, code}`) with the stores unchanged — the
  `confirmed_replace` gate, the declared-ID check and the per-key validation all
  run before the first `store.set`. Replacement is confirmation-gated because the
  same secret can be shared by several bindings and sessions, so overwriting one
  is a decision the caller has to state. An oversized body never reaches the
  handler:
  the request model rejects it and the app answers **422** with
  `{"detail": "The request has invalid fields."}` (never the rejected input,
  which is why the app owns that response). One write CAN still land alone: a
  store failure part-way through a multi-key body returns `code:
  store_unavailable` with the ids written so far in `saved_ids`, so the caller
  sees exactly which keys landed rather than a whole-body rollback. The response
  carries `{name, saved_ids, failed_ids, code}` and never echoes a value. The
  owner's RPC is a dedicated `mcp_credentials` op, never a `mcp.control` argument
  — values must not reach the slash argument, the command journal, or a request
  receipt.
- `POST /fork`: stable request_id, optional message, boundary=`next_safe`.
  The runtime refuses compaction and uses `Session.request_fork` during a turn;
  otherwise it uses `fork_session`. This is the canonical complete-history fork
  at a safe boundary, not an arbitrary transcript rewrite. The parent is unchanged.
  The child gets a new canonical ID; optional message is admitted once using the
  same UUID, never both a boot-prompt sidecar and a renderer re-submit.
- `POST /asides`: request_id, text, optional previous aside_id. Completion runs
  on the runtime but does not enter conversation history. GET `/asides/{aside_id}`
  recovers a response after HTTP loss; DELETE closes a settled panel. A continuation
  temporarily owns its prefix so two panels cannot adopt it twice.
- `POST /asides/{aside_id}/adopt`: request_id and confirmed=true. Runtime adoption
  enforces its idle guard and durable-first ordering. A latch before any await
  prevents distinct request IDs from duplicating adoption. An ambiguous failure
  is not retried under another ID. Already completed receipts replay safely.
  Asides are memory-only, bounded to 64 panels, 16 exchanges each and one hour;
  HTTP shutdown clears them. They are not canonical/durable history until adopted.
- `POST /v1/desktop/stop`: request_id, exact targets[] and confirmed=true. All
  targets are resolved before stopping any. Cold targets report already_stopped
  without starting a process. Live targets call the canonical runtime stop protocol;
  stop_requested is acknowledgement, not an invented completed-exit receipt.
- `POST /interrupt`: request_id, and deliberately no `confirmed`. It stops the
  session's CURRENT WORK — the turn that is running and the child sessions it
  started — and leaves the session, its runtime and its process alone. This is the
  rung the desktop Stop button and Esc mean; `/v1/desktop/stop` above is the KILL
  SWITCH (deny gates, dispose, release the writer lease, unpublish, exit) and keeps
  that meaning, so a control promising "stop this session's current work" must not
  be wired to it. No confirmation field, because an interrupt destroys nothing and
  requiring one would make Esc useless.

  The answer is `{status, receipt, children_running, background_jobs, replayed}`.
  `receipt` is the RUNTIME's own sentence, returned verbatim: it counts what actually
  settled, names anything that refused to die, and names any card it refused (a
  press whose only effect was clearing a question that outlived its turn reports
  `no turn was running; refused 1 waiting prompt`) — all of which only the owner
  knows.
  `children_running`/`background_jobs` are read off the follower's published roster
  AFTER the press so a surface can word its own notice without parsing prose (job
  type `task` is a subagent the interrupt did reach; `bash` is a backgrounded job it
  deliberately never touched, and the remaining lever for those is the Jobs
  surface). An `idle` answer reports what IS running rather than zeros, so a build
  this rung deliberately spares stays visible beside the status.

  `status` is `interrupted` or `idle`; `idle` is a 200 and a SUCCESS — a
  cold session is NOT engaged to answer this, and one between turns simply has
  nothing to stop. "No owner" here means the SOCKET is down, not that the viewer is
  cold: a session mid-refresh holds a live, serving runtime while its event feed
  resyncs, and it must be interrupted rather than answered `idle`.
  `interrupted` is a CLAIM that work was stopped, so the route answers
  `idle` whenever nothing would be: it reads the follower's published roster first
  and skips the owner call entirely when there is no live turn, no parked card, no
  running `task` job and no running goal loop. (A running backgrounded `bash` job is
  deliberately NOT one of those terms — this rung never touches one, so a session
  whose only live work is a build has nothing to interrupt.) A parked card with NO
  live turn is the exception that must still run, because that orphan is what the
  abort's deny-first ordering exists for; its press reports `interrupted` with a
  receipt naming the refusal. The receipt is `""` for `idle`, because there is no
  owner sentence to report and an invented one is the same overstatement the receipt
  itself is written to avoid.

  The runtime op it maps to is the existing `abort` control frame
  (`AttachedSession.interrupt` → `AttachedSession.abort` → `ServingSessionHandle
  .abort`), named `interrupt` end to end because on this surface `stop` already means
  "end the process" and a route one letter from it is a trap. Receipted
  `retry_safe=true`, so a retry after a lost response is safe and re-executed rather
  than left INDETERMINATE; the same request_id with a changed body is a 409. No
  ladder: a second press is a second interrupt, a no-op because nothing is left, and
  the control leaves the screen when `busy` goes false. Codes: 401 missing/wrong
  bearer, 403 disallowed or browser-originated Origin, 404 an unknown OR malformed
  session id (the id validator raises KeyError; 422 is reserved for the BODY's shape,
  a non-UUID request_id or an extra field), 503 an unconfigured desktop capability or
  an unreachable owner. Gated by `features.session_interrupt` — its own key, NOT a
  bump of `lifecycle`: a backend that can stop a session but cannot interrupt a turn
  must keep `/stop` working and must not be told it can interrupt, and a renderer
  that does not see the key hides its Stop control instead of firing a request an
  older backend answers 404 or, worse, wiring it to `/stop`.

`/loop <count>` uses the standing goal, max 25 iterations; `/loop <goal>` keeps an
ephemeral goal and uses the shared terminal judge protocol. The shared prompts,
verdict parser and count rules now live in `session/goal_loop.py`; terminal imports
remain compatible. The runtime waits for actual turn completion rather than HTTP
admission. It never answers gates. `--stop|stop|cancel|abort` cancels the driver and
only its own queued/active iteration; another frontend's manual turn is not cancelled
as collateral. `--clear` is the other half and is deliberately NOT a cancel: while the
driver runs it answers `a loop is running — /loop --stop to stop it first` (code
`loop_running`) and changes nothing, and when nothing runs it REPLACES the published
state with `{"status": "idle", "completed": 0}` and checkpoints it, which is what a
desktop surface dismisses and what keeps a restart from restoring the cleared run.
That refusal is a `409` carrying the sentence, like `loop_busy` — the client never has
to read a receipt to tell it from a success, which is what `POST …/commands` answers
for it (round 1 review, MINOR-4: the code rode a 200 error receipt before).
`/loop status` reads state. The canonical frontend snapshot carries
loop status/count/reason. A viewer detach does not stop/restart it; runtime teardown
cancels it, and a replaced runtime labels a retained active checkpoint interrupted
rather than automatically spending more tokens.

`/goal --clear` clears the standing goal (the bare words `clear|none|reset` still
work). It is stored nowhere and starts no turn: the receipt is the whole effect — and
it NAMES the goal it removed, because a standing goal is invisible in the UI and
there is no undo — and the argument is matched as a WHOLE, so `/goal --clear the
flaky job` remains an ordinary objective. A bare `--token` that names no flag of the
command is refused rather than stored (`/goal --stop` used to become the standing
objective), which is the one deliberate behaviour change: a goal whose text is a
single `--word` is no longer accepted.

The terminal's argument picker offers `--clear` on an empty `/goal `
argument while a goal is set, and `--stop` on an empty `/loop ` argument while that
terminal is running a loop. Both rows are `alert` rows, so one Enter FILLS the buffer
and a second runs it; the row is pre-selected, so without that gate the keystroke that
reads the standing goal would clear it.

In the terminal, `--clear` means what this host can mean by it: a loop RUNNING here is
the only loop state this host has (nothing is published), so `--clear` ends it with
the receipt `loop cleared — stopping after the current turn`, and with nothing running
it answers `nothing to clear in THIS terminal — no loop is running here`. The
owner-path refusal above is about the PUBLISHED state a detached runtime holds, which
this surface does not have.

## Provider and reporting endpoints

- GET `/v1/desktop/models?live=false|true`: ProviderController initial/cached or live
  model catalogue, selectors, connectivity and listing errors. Connectivity means
  credential availability, not proof of a successful external inference request.
- GET `/v1/desktop/usage?provider=...&live=false|true&refresh=false|true`: same shared
  ProviderController cache/account semantics as terminal usage. Reports carry age,
  quota data, unavailable/invalid-credential/partial states. Refresh requires live.
  Unsafe arbitrary provider error bodies are not returned.
- GET `/v1/auth/status`: provider accounts only, redacted identities/source,
  configured/refresh_due state and optional expiry. MCP DCR/grant rows have their
  own lifecycle and are not model-provider accounts.
- DELETE `/v1/auth/accounts/{id}`: removes that stored provider account, not sibling
  accounts or environment credentials; invalidates the shared model-listing cache.
- GET `/v1/desktop/analytics?since_ms=...&until_ms=...&session_id=...&days=...`:
  AnalyticsStore aggregate and daily series. The daily series explicitly reports
  all_sessions scope; it is not mislabelled as the optional aggregate session filter.
  The response also carries `session_names` (id -> human name) and `session_parents`
  (child id -> parent id), the store's two side attributes that a dataclass dump
  drops; a client uses them to label a per-session row and to indent a child under
  its root. `session_names` holds **only named sessions**: an unnamed one is ABSENT,
  not present-with-`""`, which is what makes a client's `name ?? id` fallback fire
  instead of painting a blank cell. Both are `{}` when the ledger carries no parent
  column, and the per-session figures stay OWN-scope — the payload hands over the
  edges so a client
  can re-partition, and never rolls a child's spend into its parent's column.
- GET `/v1/desktop/info`: the `/info` host read — `collect_snapshot(LiveState())`
  from `local_operator/info/`, run on a worker thread because the session probe
  blocks (~880 ms on macOS). It takes no parameters: there is one answer per host,
  which is why the command takes no argument at all. The snapshot is taken with **no
  session attached**, so its session-attached half (the agent counters, the subagent
  tree, the MCP probes, approval mode, skills) is reported as `null` — the unknown
  spelling, never `0`, because a host whose backend never attached a session must
  not paint "MCP 0 connected" as a fact. Those fields are facts about a SESSION and
  a client renders them from its own canonical frontend snapshot. This payload
  describes the MACHINE, which is why the panel labels it as the machine the app is
  connected to. `env.credential_keys` carries credential key NAMES only: never a
  value, length, prefix or environment value, and the read does not create the store
  it reads. Gated on `features.diagnostics >= 1`.
- GET `.../sessions/{id}/report?recent_limit=...`: one exact session's ledger report
  (`AnalyticsStore.session_report`) read in a single explicit transaction, so every
  figure comes from one WAL snapshot while the recorder may be committing behind it.
  `recent_limit` defaults to 12 and the store clamps it to 0..50. The three
  group-bys — `by_model`, `by_purpose` and `by_purpose_outcome` — are all
  **arrays** of objects, never maps: two of them are keyed by tuples JSON cannot
  carry, and the third is converted so one payload does not ship two shapes for
  one idea. `descendants_aggregate: null` and `tool_calls: null` mean the walk
  could not run and nothing was measured, which is not the same fact as zero.
  Gated on `features.diagnostics >= 1`.
- GET `.../sessions/{id}/failovers`: selected and effective runtime models plus the
  configured **default** chains. Defaults are labelled, not represented as a live
  provider's private cooldown/account routing state.
- GET `/v1/desktop/skills?session_id=...&name=...`: session-cwd discovery and optional
  known-name body via the shared internal-URL resolver. This is discoverable scope,
  not a claim that every skill was selected into the current model prompt.

## MCP controls

GET `.../sessions/{id}/mcp` is a cold-safe effective-config read. A live runtime supplies
its own manager status. Rows report source, owned scope, transport/tool count and
separate downstream_authorization=`unknown`. Tool discovery or a healthy MCP
transport does **not** prove Google Workspace account authorization.

POST the same path accepts the closed `MCPControl` schema:

- `add`: name, scope global/project, either command+args[] or url; optional env,
  headers and oauth boolean. Env/header values must be `${NAME}` references,
  resolved at connect time from the encrypted secret store; a legacy
  `credentials.env` value is used ONLY when the reference is definitively absent
  from that store (never on a denied, locked, corrupt or empty entry). A reference
  that cannot be resolved fails the connect naming the key; it never
  reaches the server as text. A doubled `$` (`$${HOME}`) escapes one to literal
  text. URLs reject inline credentials, query and fragment.
  Command arguments remain an array; no shell evaluation or whitespace splitting.
  Store secrets separately, through `POST /mcp/credentials`.

  Each server row also publishes `secret_refs: [{id, bindings: [{field, key}]}]`
  — the reference IDs its PRISTINE config declares and the destinations they are
  bound into, deduped by ID. This is metadata only: no template text, no literal
  fragment and no value. `environment_keys`/`header_keys` remain as INFORMATIONAL
  map keys and are never secret IDs — writing their values as credentials is the
  bug this metadata exists to prevent. `probe` additionally answers
  `secret_refs`, `credential_state: [{id, source: encrypted|legacy|missing|
  unavailable}]` and `key_submission_supported`; a server declaring no reference
  gets an honest setup sentence rather than a guessed field binding.
- `remove`: name, exact owned scope, confirmed=true. The existing ownership resolver
  refuses removal of foreign imported definitions and does not shadow them.
- `reload`, `connect`, `disconnect` use the session's existing manager. Disconnect
  requires confirmation. Read status to distinguish connecting from connected.
- `probe` resolves actual transport OAuth capability through existing core code.
  A statically incompatible stdio/API-key server reports false; otherwise list
  metadata remains unknown until probed. Do not offer HTTP OAuth as stdio setup.
- `login|logout|reauth` starts a runtime operation; logout and reauth require explicit
  confirmation. `status|cancel` takes operation_id. Operations are bounded, one grant
  at a time per runtime, timeout after five minutes, and keep the runtime resident.
  Credential deletion during reauth is reported even when the later login is
  cancelled. No grant tokens cross HTTP. Core OAuth owns callbacks, refresh locks
  and auth.db; desktop does not add a second store.

Rows also supply a session-prompt setup action to inspect server-supported setup
when downstream account authorization is server-specific. It is an offer for the
user to submit normally with ordinary gates, not an automatic setup-tool call.
Legacy Google token values are deliberately retained for user scripts.

## Radient: narrow proxy, not another authentication authority

POST `/v1/desktop/radient` selects one of 25 closed operations: account/prices,
credits/usage, provision/application.create, agent catalogue/detail/CRUD,
like/favourite/count controls, comments/CRUD and account.agents. Paths are assembled
server-side from bounded identifiers. Query/payload keys are allowlisted per
operation. Mutations require stable request_id; DELETE additionally requires
confirmation. Redirects and oversized/upstream error bodies are refused.

The backend resolves and refreshes the Radient credential through AuthStore.
Provisioned application keys are stored centrally and removed from the response.
There is no token getter, token exchange/refresh proxy, arbitrary URL or arbitrary
header operation. Reads of public prices do not require a Radient login. UI-owned
OIDC/refresh/keytar effects must be removed by the frontend implementation, not run
alongside this path.

Every refusal of that proxy answers `{"code", "message", "details"}` in `detail`,
and `code` is what separates the remedies: `radient_no_credential` (nothing is
stored) and `radient_credential_refused` (Radient refused this account's sign-in —
including a grant the store knows is dead, which carries
`details.reason = "grant_invalid"`) both mean "sign in to Radient again", while
`radient_upstream_failed` means "retry" (`details.reason` distinguishes a transient
refresh failure from an outage, and `details.upstream_status` carries the answer
when there was one). A refusal from the desktop plane itself — this app's own
bearer — carries no `radient_` code, and that absence is how a client tells
"re-pair the app" from "sign in again". A bearer the store already considers due
for a refresh is never spent upstream: the proxy answers the classified refusal
rather than relaying Radient's 401 for a token it knows is stale.

The old Google integration UI writes GOOGLE_ACCESS_TOKEN, GOOGLE_REFRESH_TOKEN and
GOOGLE_TOKEN_EXPIRY_TIMESTAMP via `use-oidc-auth.ts`; no builtin backend reader or
Gmail/Calendar/Drive client consumes those keys. The only other UI references are
OAuth handlers, credential labels and badges. That legacy credential-acquisition
UI is replaced by actual configured MCP integration/grant management, **not** by
inventing a Radient integration endpoint or claiming Radient console scopes grant
Workspace access. Preserve stored keys; explain the separate MCP/server setup.

## Legacy Radient client compatibility

`providers/radient_credentials.py` is the shared compatibility resolver for legacy
server clients and the CLI push/delete paths. It uses canonical AuthStore precedence
(OAuth, login key, environment/legacy fallback and remaining core tiers), never a
copied credential file or another refresh store. Server callers reuse the existing
DesktopAuth store and its refresh lock; the synchronous CLI adapter is lazy and
uses the same resolver. Legacy key values remain untouched. An explicitly configured
foreign/gateway endpoint receives only its previous dedicated legacy key, never a
centrally signed-in Radient bearer.

The legacy model catalogue, speech, transcription and agent ZIP-upload consumers
now require the same bearer/Origin boundary in managed mode before using central
credentials. Standalone unmanaged servers retain their legacy access model.
The JSON transport includes `legacy.models` and `legacy.agent.upload`; the frontend
must also supply authenticated binary speech and multipart transcription relay as
part of its native media/stream integration. The JSON-only relay cannot be used as
though it already carries those binary media responses. This is an explicit remaining
frontend transport obligation, not a credential sync workaround.

`test_desktop_legacy_radient.py` exercises real legacy HTTP clients and the CLI delete
command against a threaded fake upstream: an AuthStore-only credential drives real
speech bytes, model listing and ZIP upload, while the old credential file stays empty.
It also checks managed auth/origin failures and that reflected upstream credentials
are suppressed by the existing HTTP error-body policy. Unit tests pin precedence,
custom endpoint isolation, legacy fallback and a single refresh across concurrent
server readers. The CLI retains its historical endpoint joining; no new remote API
compatibility is asserted by the credential change.

## Verification and remaining UI gate

`tests/e2e/test_desktop_controls.py` drives actual HTTP + canonical Session,
ServingSessionHandle/RuntimeServer/AttachClient, command census, secret lifecycle,
loop count/goal/cancellation, aside/adoption, fork, selected stop and real stdio MCP.
External model replies alone are scripted. Its interrupt case reproduces the reported
Stop-button defect (the old `command: "stop"` body answers a `native_action` while
the turn keeps streaming), then interrupts a real streaming turn, replays the receipt,
runs a SECOND real turn on the same session to show the session outlived it, answers a
cold session `idle` without creating a runtime record, and asserts the 401/403/404/422
shapes. Existing desktop session/spawn tests cover real detached subprocess admission,
stale gate answers, replay/reconnect, watch leases and HTTP restart.

`test_desktop_mcp_oauth.py` drives a real local HTTP MCP/OAuth issuer, real callback
listener, auth store and grant cancellation. Consent is simulated by a fixture HTTP
redirect, not a real browser or Google authorization. `test_desktop_radient.py`
drives a local fake upstream, 25 operations, real central refresh, key-storage side
effects and negative cases. It is not proof of real third-party account access.
Live third-party quotas/catalogues still depend on external credentials/services;
the adapter reuses their already-tested ProviderController authority.

The transport contract tests drive real loopback HTTP with an Electron IPC fixture,
not a native app. No renderer rendering, clipboard, updater, browser OAuth UX,
notifications, stream IPC delivery, screenshot or full desktop-parity acceptance is
claimed by this backend slice. Each matrix row's UI acceptance remains an explicit
frontend/design/QA gate before release.
