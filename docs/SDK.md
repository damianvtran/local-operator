# The Python SDK (`local_operator.sdk`)

`local_operator.sdk` is the programmatic embedding surface over the session
machinery: the same composition root (`session_factory.create_session`), the
same engagement router (`engage_runtime` / `spawn_owned_session`) and the same
session object every front end drives. `lop exec`, the TUI, the desktop server,
the mobile relay and the benchmark apparatus all consume those objects; the SDK
exists so a script joins them — instead of hand-building an
`argparse.Namespace` and its own tool list, and quietly missing everything the
harness already does.

**The rule for editors, stated once here and in the module:** the SDK may adapt
*shape*, never *semantics*. Compaction, prompt caching, failover, approvals,
delegation, skills, guides and MCP are the session's behaviour and reach SDK
callers exactly as they reach every other surface; a behavioural change belongs
in `Session` / `harness` / `session_factory`, where every consumer sees it. Do
not add a loop, a tool dispatcher, a subagent runner or a second
serialization of `AgentEvent` here.

## Quickstart

```python
import asyncio
from local_operator.sdk import ApprovalPolicy, SessionRoots, SessionSpec, events, open_session


async def main() -> None:
    roots = SessionRoots(  # REQUIRED: explicit roots, no ambient default
        config_dir="/scratch/ep/home/.local-operator",
        agent_home="/scratch/ep/home",
        cwd="/scratch/ep/work",
    )
    spec = SessionSpec(
        hosting="openrouter",
        model="deepseek/deepseek-v4.1-flash",
        approvals=ApprovalPolicy.declared(["read", "grep", "glob"]),
        name="nightly audit",
    )
    async with open_session(spec, roots=roots) as session:
        stream = events(session)  # subscribe FIRST — events are not replayed
        await session.prompt("Summarize the repository")  # awaited to the turn's outcome
        async for event in stream:
            print(event.type)
            if event.type == "agent_end":  # this prompt's terminal event
                break  # the stream itself never ends — end it explicitly
        await stream.aclose()  # unsubscribe once you are done


asyncio.run(main())
```

The event JSON projection for line-oriented consumers is
`local_operator.headless_print.printable_event` — the same mapping `lop exec
--json` prints. The SDK does not invent a second serialization.

## Verbose diagnostics

Set `verbose=True` to log tool calls, output validation, turn usage, and
terminal status at `INFO`. The SDK does not configure logging for your app.
For example, in a standalone script (using the `spec` and `roots` above):

```python
import logging

logging.basicConfig()  # In an existing app, use its configured handlers.
logging.getLogger("local_operator.sdk").setLevel(logging.INFO)

async with open_session(spec, roots=roots, verbose=True) as session:
    await session.prompt("Summarize the workspace.")
```

A tool status of `returned` means the tool did not report a tool-level error;
it does not imply that a subprocess it ran exited successfully.

### Preserve a warm built-in tool prefix on owner reopen (opt-in)

Pass `warm_tool_prefix=True` to `open_session` on both the original owner and
its reopen. The current implementation is intentionally limited to a confined
`world_root`, the default `ApprovalPolicy.refuse()`, an undeclared tool inventory
with no MCP servers configured, and Bedrock's native Claude Messages route.
It can preserve only the shipped `library` schema after that schema reached a
completed turn that reported a five-minute cache write. A live, unchanged
inventory must reproduce the exact ordered tool array; otherwise the new owner
publishes as usual. There is no extra readiness wait and no cached definition,
execution permission, or approval is restored. Reopening alone does not extend
the hint's lifetime, and a matching prefix does not guarantee a provider hit.

## Entry points

| Entry point | What it does | Exec-parity reference |
| --- | --- | --- |
| `open_session(spec, roots=..., mode="own"\|"attach")` | `own`: builds the session in this process (like `lop exec` foreground) and yields it for use inside an `async with`. `attach`: the viewer path for an id whose runtime is **already live**; a cold id is refused with a remedy. An attach spec carries `resume` only — owner-side fields (team/profile/tools/name/goal), non-default `approvals` and `output_*` enforcement are refused, not silently inert. | `session_factory.create_session` |
| `spawn_session(spec, roots=..., errand=...)` | Runtime-hosted and survives the caller: mints a viewer-style id (`uuid4().hex[:12]`), warms a detached runtime with the spec's birth sample, then delivers the errand over the one engagement router. | phone/TUI spawn (`engage_runtime`) |
| `deliver(session_id, roots=..., errand=...)` | One engagement for an existing session (`id` or `@latest`), cold or live. | `lop send` / mobile engage |
| `events(session)` | Async iterator over `session.subscribe()`; `aclose()` (or `async with`) unsubscribes. | `exec --json`'s stream |

Errands are the runtime's typed vocabulary, re-exported:
`PromptErrand` (a user turn), `SteerErrand`, `PeerMessageErrand`, `WakeErrand`,
`WarmErrand`. `spawn_session`/`deliver` return the router's `EngageOutcome`
(`session_id`, `detail`, `spawned`, `duplicate`).

Lifecycle verbs are the session's own, unchanged: `abort()` ends a turn,
`dispose()` ends an owner (the `async with` does this for you), a viewer's
`interrupt()` is turn-scoped. `request_stop` — the user's kill switch — is
deliberately **not** exposed: only a person ends a session.

## What a spec carries

`SessionSpec` mirrors `lop exec`'s options. The eight session-namespace fields
(`hosting`, `model`, `agent_name`, `agent_id`, `yolo`, `train`, `resume`,
`workstream`) are pinned **field-for-field** against
`exec_mode._make_default_session_factory`'s namespace by a parity test — the
same construction, not a copy of it. On top:

| Field | Meaning |
| --- | --- |
| `birth_effort` | Construction-time reasoning level (clamped by the factory, like `spawn_owned_session`'s documented extra). |
| `team`, `profile` | Post-open attachments, resolved against **this** root by the same helper exec uses (`exec_startup.resolve_startup`) and attached through the same `Session` methods (`attach_team`, `attach_agent_profile`). |
| `tools` | The run's whole reach — exec `--tools` semantics, one-way for the session's life; a second declaration may only tighten. Excluded tools are unreachable by name. |
| `approvals` | See the table below. |
| `name`, `goal` | Conversation title and standing goal. |
| `output_format`, `output_schema`, `output_retries` | Enforce the final response's format (and optionally a schema) for every turn — the same contract `lop exec --output-format` installs, applied through the same post-open `Session.set_output_contract`. See [Enforcing the final response contract](#enforcing-the-final-response-contract). |
| `mcp_ready_timeout_s` | How long the first `prompt()` of an in-process session waits for MCP servers still connecting (default `30.0` s; `None` opts out). See [Required-tool readiness](#required-tool-readiness--the-first-prompt-waits-by-default). |
| `notifications` | A *spawned* runtime is silenced by default (the harness's "nobody is watching" posture); `True` lets it inherit the launcher's posture. In-process sessions leave notification policy to the host process. |

Not carried, deliberately: `--effort` (a runner-level knob applied after
construction; use `birth_effort` for the construction-time level), `--loop` /
`--loop-goal` (exec's continuation mechanism; an SDK caller drives turns
itself), `--clear-goal` (adjusts a resumed run rather than starting one).

## Enforcing the final response contract

`SessionSpec.output_format` (with optional `output_schema` and
`output_retries`) makes the session's harness validate each turn's terminal
assistant text, and retry inside the same turn when it does not conform — the
same `Session.set_output_contract` method `lop exec --output-format` installs,
through the same validator (`OutputContract`). `None` (the default) leaves
every turn byte-identical to a session without it. The `exec` docs own the
per-format strictness table and the extraction rules; the schema spelling here
is a TYPE or a MAPPING rather than a file path:

```python
from pydantic import BaseModel
from local_operator.sdk import SessionSpec, SessionRoots, events, open_session


class Invoice(BaseModel):
    total: int
    currency: str


spec = SessionSpec(
    hosting="openrouter",
    model="deepseek/deepseek-v4.1-flash",
    output_format="json",
    output_schema=Invoice,          # model class, dataclass, TypedDict, raw JSON Schema, …
    output_retries=1,               # 0-5; default 2 (three attempts per turn)
)

async with open_session(spec, roots=roots) as session:
    stream = events(session)
    await session.prompt("Extract the invoice")
    async for event in stream:
        if event.type == "output_validation" and event.ok:
            invoice = Invoice.model_validate_json(event.payload_text)  # the validated span
        if event.type == "agent_end":
            if event.error:  # exhaustion is a turn error, like any provider error
                raise RuntimeError(event.error)
            break
```

Failure semantics are the SDK's ordinary turn semantics: an exhausted budget
ends the turn with `agent_end.error` of
`final response did not satisfy the output contract (…) after N attempts: <reason>`
and `session.last_turn_outcome == "error"` — `await session.prompt(...)` still
resolves (a turn error is an event, not an exception, exactly as every provider
error). A successful check is visible as an `output_validation` event with
`ok=True` and `payload_text` (the exact span that validated);
`local_operator.sdk.decode_output` re-decodes that span with the same rules if
you only have the text.

Refused rather than silently inert: `output_*` on an `attach` spec (the viewer
cannot install owner state) and on `spawn_session` (post-open state has no
channel to a new runtime child). The composition that works is the documented
one: `open_session(..., mode="own")` → `dispose` → `spawn_session` the same
id, or drive the session in-process.

## Typed completion evidence

An owning `Session` can independently check a completion claim with
`await session.verify_completion(...)`. This is an explicit embedder action,
not a tool the worker can invoke or a second provider loop. Its normal
read-only child, request/inspection budgets and wall deadline still apply.

```python
from local_operator.sdk import VerifierEvidence, VerifierEvidenceItem
from local_operator.harness.types import ImageContent, TextContent

proof = VerifierEvidence(
    items=(
        VerifierEvidenceItem("current-screen", ImageContent(
            data=screenshot_base64, mime_type="image/png",
        )),
        VerifierEvidenceItem("observed-text", TextContent(text=observed_text)),
    ),
    required_modalities=frozenset({"image"}),
)
verdict = await session.verify_completion(
    request=original_request,
    claim=worker_completion_claim,
    typed_evidence=proof,
    host_tools=read_only_target_tools,
)
```

The envelope carries actual image blocks through `Session.prompt(images=...)`,
not base64 in prompt text or tool details. IDs identify supplied content, never
paths or URLs the checking model may open. Successful counted host inspections
receive additional `inspection-N` evidence IDs; receipts are fresh for each
verification, even when the caller reuses the envelope. The evaluation adapter's
`inspect_command` also accepts **only** `artifact_id` (mutually exclusive with
`command`/`timeout_s`) to return an already-registered frame as image content.
It gains no screenshot/rendering/browser or arbitrary-file capability.

With typed evidence, the child must return one strict JSON assessment containing
`requirements`, each with `requirement`, `status` (`MET`, `UNMET`, `UNVERIFIED`),
`modality` (`text`, `image`), `evidence_ids`, and `findings`. The outward
`VerifierVerdict` remains unchanged: `UNMET` aggregates to `FAIL`; unverified,
malformed or unavailable/mismatched evidence aggregates to `INCONCLUSIVE`;
`PASS` requires valid `MET` rows citing evidence of their declared modality and
covering every required modality. This validates the requirements the model
identified, **not** a deterministic guarantee it enumerated every requirement.
`UNMET:` findings continue through the existing worker repair challenge.

Required images use the ordinary resize/request-fitting path, but cannot be
silently replaced by text: missing/corrupt media, a fitting pass that sheds a
required frame, provider rejection, or a serving model without explicitly
stated vision support yields an inconclusive check. Capability is rechecked
at actual failover dispatch. There is no extra model call or increased cap.
Callers omitting `typed_evidence` keep the existing verdict-line contract;
ordinary sessions do not run the typed decoder or required-image validation.

## Approval policies

| Preset | Gate behaviour | Exec analogue |
| --- | --- | --- |
| `ApprovalPolicy.refuse()` **(default)** | Typed refusal (`ApprovalUnavailableError`) — never a silent `False` rendered as "user denied". | default non-tty run |
| `ApprovalPolicy.auto()` | Every gated call approved. | `--yolo` |
| `ApprovalPolicy.declared([...])` | The named tools stand as their own approval, inside a reach bound to exactly them (excluded tools are unreachable, and delegated children inherit the bound). | `--tools` where nobody can be asked |
| `ApprovalPolicy.callback(fn)` | `fn` decides, exactly as a full front end's `set_approval_handler`. | a full-screen front end |

Where the SDK is *stricter* than exec, by design: exec derives the
declared-tools stand-as-approval from a **terminal** (`not control and not
stdin.isatty()`). An SDK process has no terminal to consult and its stdin says
nothing about whether a person can be asked, so the stand is opt-in here — the
caller says `declared([...])` — and `refuse()` keeps a declared inventory's
write/exec calls gated. The mapping is pinned by tests.

`ask` questions are not installed by any preset (mirroring exec's headless
paths, which install none — the `ask` tool is simply absent until a caller
supplies a surface on the session object).

## Isolation — the defaults are the safe ones

A programmatic session must not be able to "just happen" to target the
operator's real store, and the SDK makes that unrepresentable by accident:

* **Roots are required and explicit.** `SessionRoots(config_dir, agent_home,
  cwd)`; there is no default. The facade scopes `LOCAL_OPERATOR_CONFIG_DIR` /
  `LOCAL_OPERATOR_HOME` to those roots for construction (and for the whole
  `async with` body in own mode), so every lazy resolver sees the same root.
* **The resolved roots are asserted, not assumed.** Construction fails loudly
  if the environment resolves anything but the declared roots.
* **The uid-default roots are refused** unless the caller passes
  `allow_ambient=True` — a greppable, deliberate opt-in for single-machine
  scripts. "Default" is answered against the uid's **passwd home**, never
  `$HOME` (an isolated run's `$HOME` lies — the same reasoning as the browser
  bridge's supervisor naming).
* **The cache is checked by its real resolver.** The model-listing cache
  derives from `$HOME` independently of the two overrides; if it resolves
  outside the declared roots while `HOME` is this uid's real home, the run is
  refused with the remedy (redirect `HOME` — the reliable method, see
  `AGENTS.md` §Isolating a run).
* **Roots must be durable** — a store under `/tmp`, `/var/tmp` or `$TMPDIR`
  can be purged mid-run (it deleted a pilot's rescue root once). Opt out with
  `allow_volatile=True`.
* **One root per process.** `open_session` (and `spawn_session`/`deliver`)
  refuse a second, different root while another is live; `allow_multi_root=True`
  is the deliberate escape for a migration tool or a test harness.
* **Spawned children get the root explicitly.** The child's environment is
  built from `SessionRoots` (`HOME` = agent home, the two overrides, `CMUX_*`
  and `LOP_*` stripped, notifications silenced unless opted in) instead of
  relying on inheritance surviving a launchd hop.
* **The agent-shell policy is the CLI's.** A session-creating call from inside
  an agent's shell is refused unless the shell holds the delegation allowance
  (or the documented QA escape is set) — and the session it opens anyway is
  stamped by `session_factory._prepare`, the one place exec's stamp is written,
  so it stays out of the operator's picker, sidebar and phone list.

Prove inheritance the way the apparatus does: print the resolved store path
(`local_operator.paths.config_dir()`), not the config that should produce it.

## Localized startup refusals — source-revision capability

At source `641a392040bf4056361228cfc3d3ad978fced3e8`,
[#2317](https://github.com/damianvtran/local-operator/pull/2317)
(merge `7c29e7c3f2c0507f21bb3ca566cf6901b110c113`) extracts exactly **four**
[English `wire.errors.sdk_startup` keys](../local_operator/i18n/catalogues/en/wire.errors.sdk_startup.json):
`attach_hosting_unconfigured`, `attach_runtime_missing`, `profile_attach_failed`
and `tool_inventory_mismatch`, with context and generated Python functions.
This does not translate every SDK validation/refusal or add a typed exception API.

Only these refusal paths lazily resolve language, **after public root isolation**:
they reuse the manager scoped to the explicit `SessionRoots`, then follow
config → `LOP_LANG` → OS → `en` and shipped eligibility. Missing namespace/key
copy uses English source and formatting. Internal `Msg` values are rendering
inputs; callers still receive a plain, **one-argument `SessionSpecError`** and
read its human-readable text with `str(error)`. No public `{code, params, text}`
exception envelope, new exception fields or locale parameter is introduced.

Spec/input refusals (`wire.errors.sdk_spec`, raised by `SessionSpec`,
`ApprovalPolicy`, `open_session` and `spawn_session` validation) render in the
process's **active** locale, which `lop` moves at boot and on a language change.
They never read configuration to choose a language — they fire before root
isolation — so a bare SDK embedder that never boots a locale keeps the English
source byte-for-byte. The exception shape is unchanged (one string argument).

This describes that source revision, not a promise about an older installed SDK.
The measured 2026-10-13 UTC installed 0.70.10 snapshot predates this slice; verify
installation contains it before relying on localized SDK refusals. No provider
request is needed to inspect version/source identity. See [the i18n HOWTO](I18N.md)
for the distinct source, tag, publication and installation observations.

## Required-tool readiness — the first prompt waits by default

MCP discovery is asynchronous: servers deferred past the startup gate connect
on background continuations, so a prompt sent right after `open_session` can
win the race — measured against a 1.5 s stdio server in fresh roots, the first
request left 0.36 s after `prompt()` with the server still `connecting`, and
the model's `read mcp://slowfix/search` answered "Unknown tool".

**So the first `prompt()` of an in-process session (`mode="own"`) waits,
bounded, for every configured MCP server that is still connecting** — not
only for declared tools. The wait ends the moment nothing is left connecting;
servers that already settled (`auth-required`, a crashed command, one parked in
reconnect backoff) never hold it, and a server that settles as failed is
logged and the prompt proceeds without its tools. Later prompts do not wait
again once the first one got through. Cancelling the `prompt()` task ends the
wait at once, and `session.abort()` within a quarter-second poll; either way
discovery keeps running and the next `prompt()` meets the wait again.

```python
# Default: wait up to 30 s (the measured first-turn bound; cold stdio servers
# took median 0.68-0.85 s, max 3.48 s).
spec = sdk.SessionSpec(hosting="anthropic", model="claude-sonnet-4-5")

# A tighter (or looser) bound; 0 means "must already be connected".
spec = sdk.SessionSpec(hosting="anthropic", model="claude-sonnet-4-5", mcp_ready_timeout_s=5.0)

# Opt out: the first prompt is sent at once and may race discovery
# (declared tools included); gate by hand with wait_for_tools() if needed.
spec = sdk.SessionSpec(hosting="anthropic", model="claude-sonnet-4-5", mcp_ready_timeout_s=None)
```

A server **still connecting when the bound runs out** fails the prompt with
`McpServersNotReady` (from `local_operator.session.declared_mcp_ready`)
*before anything is sent*: `servers` maps each still-connecting server to its
status, `connected` names the servers that made it, `elapsed_s`/`timeout_s`
say how long it waited. Discovery is not cancelled, and the next `prompt()`
meets the gate again rather than racing. A declared tool whose server failed
keeps raising `DeclaredMcpToolsUnavailable`, as before. The setting applies to
`open_session(mode="own")` only: `mode="attach"` and `spawn_session` refuse a
non-default value, since neither runs the turn loop in this process.

For a specific tool, owning sessions also expose a typed, bounded gate instead
of leaving each caller to hand-roll a manager wait plus an inventory read:

```python
async with sdk.open_session(spec, roots=roots) as session:
    readiness = await session.wait_for_tools(
        ["mcp__episode_actions_apply_actions"], timeout_s=30.0
    )
    if readiness.status != "ready":
        raise RuntimeError(
            f"action tool not reachable: {readiness.missing} ({readiness.status})"
        )
```

`status` is `ready` (every name is in the live enforced inventory — already,
or after one bounded discovery wait), `pending` (still missing while
discovery is in flight) or `missing` (discovery settled without it, or no
discovery manager exists). The result also partitions the request into
`present`/`missing` name tuples and reports whether discovery had settled.
Naming a lazy MCP tool here activates it, exactly as a declaration does: a
connected server's tools stay out of the enforced inventory until something
selects them, so a named one is granted its schema (within the session's
declaration, never past it; never in a delegated child) instead of reading
`missing`.

Details that matter: already-reachable names — built-ins included — return
ready immediately without waiting on unrelated slow servers; a timeout
reports `pending` and never cancels discovery, so a later fresh call can
still read ready; failed, misspelled or declaration-excluded names read
`missing` and never widen the inventory; a manager failure never reads ready;
caller cancellation propagates and leaves discovery alive. This is
point-in-time readiness, not a lifetime guarantee against later disconnects.
Attached viewers cannot wait on another process's discovery, so the API
lives on `OwningSessionProtocol` only (see
`local_operator/session/tool_readiness.py`).

## Scope of this first release, stated plainly

Published **additive-stable** (in-repo; no PyPI split, no semver promise).
Attachable later: a warm session id (`spec.with_resume(id)`) opens with
`mode="attach"` once a runtime is live. Deliberately deferred, and refused
loudly rather than ignored (an inert setting a caller believes in is the
failure shape this surface refuses to have):

* **attach auto-spawn** — a cold id gets a remedy, not a spawn; use
  `spawn_session`/`deliver` first.
* **post-open state on `spawn_session`** — `team`, `profile`, `tools`, `name`,
  `goal`, non-default `approvals`, `yolo` and the `output_*` enforcement fields
  have no sanctioned channel to a *new* runtime child. The composition that
  works: `open_session` (attach what you need) → `dispose` → `spawn_session`
  the same id; resume restores the attachment sidecars.
* **`spec.resume(id)` as a method** — the field keeps the name for exec parity;
  the helper is `spec.with_resume(id)`.

## The parity check this surface answers to

The falsifiable core of "the same mechanism as `lop exec`": after the
follow-on work, `exec --json` and an SDK session built from the same spec must
produce **the same tool surface, the same approval semantics, and event
streams of the same shape** (modulo transport). The construction-time half is
pinned today — the spec→namespace parity test, the tool-surface comparison
against `tools.registry.create_tools` for the same allow-list, and the
approval-policy mapping table; the end-to-end `exec --json` comparison lands
with the benchmark pilot (PR 2), which is also the first real consumer.
