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

## Entry points

| Entry point | What it does | Exec-parity reference |
| --- | --- | --- |
| `open_session(spec, roots=..., mode="own"\|"attach")` | `own`: builds the session in this process (like `lop exec` foreground) and yields it for use inside an `async with`. `attach`: the viewer path for an id whose runtime is **already live**; a cold id is refused with a remedy. An attach spec carries `resume` only — owner-side fields (team/profile/tools/name/goal) and non-default `approvals` are refused, not silently inert. | `session_factory.create_session` |
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
| `notifications` | A *spawned* runtime is silenced by default (the harness's "nobody is watching" posture); `True` lets it inherit the launcher's posture. In-process sessions leave notification policy to the host process. |

Not carried, deliberately: `--effort` (a runner-level knob applied after
construction; use `birth_effort` for the construction-time level), `--loop` /
`--loop-goal` (exec's continuation mechanism; an SDK caller drives turns
itself), `--clear-goal` (adjusts a resumed run rather than starting one).

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

## Scope of this first release, stated plainly

Published **additive-stable** (in-repo; no PyPI split, no semver promise).
Attachable later: a warm session id (`spec.with_resume(id)`) opens with
`mode="attach"` once a runtime is live. Deliberately deferred, and refused
loudly rather than ignored (an inert setting a caller believes in is the
failure shape this surface refuses to have):

* **attach auto-spawn** — a cold id gets a remedy, not a spawn; use
  `spawn_session`/`deliver` first.
* **post-open state on `spawn_session`** — `team`, `profile`, `tools`, `name`,
  `goal`, non-default `approvals` and `yolo` have no sanctioned channel to a
  *new* runtime child. The composition that works: `open_session` (attach what
  you need) → `dispose` → `spawn_session` the same id; resume restores the
  attachment sidecars.
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
