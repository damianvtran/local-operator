# Design: the standard artifact-generation abstraction (`local_operator.artifacts`)

Status: architect proposal — media wave-2, requirement 3 (standard artifact-generation
interface; provider breadth). Base: `c5dee91c9c` (`feat/media-provider-breadth`,
`~/local-operator-worktrees/media-wave2-providers`). Author: architect report,
2026-10-09 (session 4dc01d842c67, lopdev). **Do not implement video in this wave** —
this document names the seams video will use and nothing more.

Companion reading (all read for this doc):

- `docs/design/image-generation.md` (this worktree) — the frozen image v1 design
  (D1–D12, §3 cascade, §4 cancel, §5 result contract).
- `~/radient-ml/agent-server/docs/MEDIA-PROVIDERS.md` §5.1–§5.10 — the hub-side
  provider-seam RFC (merged, PR #91).
- `~/radient-ml/agent-server` branch `spike/media-provider-seam` —
  `internal/services/media/provider.go` (the spike vocabulary: `State`, `Ref`,
  `Handle`, `CancelOutcome`, `ErrorClass`, `Provider`) and
  `internal/services/media/altprovider/altprovider.go` (the proof that a vendor
  with a totally different wire sits behind the same seam).
- Current code, all cited below: `local_operator/imagegen/{__init__,cascade,rungs,availability,errors,media}.py`,
  `local_operator/tools/image_tool.py`, `local_operator/tools/registry.py`,
  `local_operator/providers/{registry,login_catalog,clients}.py`,
  `local_operator/harness/loop.py`, and the four progress-consuming surfaces
  (TUI `local_operator/tui/imagegen.py`; relay `local_operator/mobile/web/src/lib/image-gen.ts`;
  desktop `~/local-operator-ui` `src/renderer/src/features/chat/canonical/image-gen-card-model.ts`;
  native `~/local-operator-mobile` `src/features/session/imagegen.ts`).

## 0. The problem, as found (not as assumed)

The image lane works today (design D1–D12, `image-generation.md`) and its
behaviour is pinned by tests. But the code has **no artifact-generation abstraction**:
every token it owns is named for its one artifact kind, and a second kind (video)
or a new provider rung has nowhere to plug in that does not mean editing the
image lane's furniture.

**Goal.** One job interface — submit / progress / cancel / steer / restart — that
`generate_image` rides today with **zero behaviour change**, that the wave's four
new provider rungs attach to (each an adapter on the same seam, proving it across
three transports), and that video and other artifact kinds extend later without
forking the walk, the progress payload or the cancel path.

What actually exists (`c5dee91c9c`):

- **Kind-named vocabulary** — `ImageRoute`, `ImageAttempt`, `ImageOutcome`,
  `ImageRouteResolution`, `ImageGenerationCancelled`, `ImageGenerationUnavailable`
  (`local_operator/imagegen/__init__.py:47-140`, `cascade.py:82-109`). A few
  tokens are *already* kind-neutral by accident: `AttemptOutcome`
  (`__init__.py:62-66`), `RungAvailability` (`:69-79`), `MediaAsset`
  (`:110-119`) — `MediaAsset` even carries `duration_s` (`:119`).
- **The walk** — `run_image_cascade` (`cascade.py:351-488`): fixed-order rung
  iteration, per-rung budget `min(240s, remaining)`, overall 300s deadline
  (`:70-72`, `:379-396`), fail-forward on every rung failure except user
  cancellation (`:418-420`) and local validation, skipped attempts for budget
  exhaustion (`:383-395`) and `RungSkipped` (`:442-451`), attempt records, and
  `ImageGenerationUnavailable` carrying resolution + attempts (`:483-485`).
- **Rung executors** — `run_radient`/`run_fal`/`run_openai` (`rungs.py:595,807,991`):
  httpx-async only (D6, `rungs.py:3-9`), each owns its transport (Radient hub
  poll, FAL queue poll, OpenAI single sync request) and fills a shared
  `CancelHandle` (`rungs.py:241-271`) only where a provider job exists; the
  best-effort cancel dispatches on two provider arms (`rungs.py:1127-1219`).
- **The progress wire** — `progress_details()` emits one canonical payload
  (`rungs.py:393-441`; keys `tool_name`, `stage`, `provider`, `model`,
  `elapsed_s`, `num_images`, `queue_position`, `progress_fraction`, `log_lines`,
  `error`, `error_type`; every key present, `None` for absent provider data;
  `progress_fraction` deliberately never synthesised `:418-421`) via
  `emit_progress` (`:376-390`), consumed by FOUR surfaces through their own
  one-reader adapters (see §6).
- **The tool** — `generate_image` (`image_tool.py:106-169`, registered at
  `tools/registry.py:116` and `:158`): write-tier approval, `interruptible=True`
  (`:159-166`), bounded best-effort cancel on both cancellation spellings
  (`:456-484`), result contract via `cache_media` (`:516-603`). Its description
  is pinned whole at **186 chars / 117-char first sentence**
  (`tests/unit/tools/test_image_tool.py:89-93`).
- **The provider registry** already carries the capability vocabulary the seam
  needs: `CAPABILITY_VOCABULARY = {"chat","tts","stt","image","video"}`
  (`providers/registry.py:78`), `media_only` rows for FAL (`:898-916`) and the
  Radient hub's `{"chat","image","video"}` (`:793`) — and *rows for every new
  rung this wave adds already exist*: `openai` OAuth (`:455` ff., ChatGPT
  Plus/Pro), `openai-key` (`:723`), `google` (AI Studio key, `:650-664`),
  `xai` + `xai-oauth` (both store under `xai`, `:567-581`), `openrouter`
  (`:689`). The Codex backend URL is already a harness constant
  (`providers/clients.py:2374`, `CODEX_RESPONSES_URL`).

And what is missing for the wave:

1. **One interface for async artifact jobs** — submit / progress / cancel /
   steer / restart — currently implicit in image-shaped functions; video and
   the new sync/streamed rung shapes must ride it without forking it.
2. **Provider breadth** — research (manager note, 2026-10-09) landed the wave's
   rung set: OpenAI subscription (Codex backend, SSE), Google Gemini API key,
   xAI (key or OAuth), OpenRouter (reported cost) — on top of Radient/FAL/OpenAI-key.
3. **A provider-neutral rung protocol** that holds across three transports the
   wave now needs: **poll-based** (Radient, FAL), **single request/response**
   (OpenAI-key today; xAI, OpenRouter, Google), and **SSE-streamed**
   (OpenAI-subscription's Codex backend).

## 1. Options and recommendation

### 1.1 Options considered

- **A — new neutral package `local_operator/artifacts/`, imagegen becomes its
  first adapter (recommended).** Kind-neutral tokens, the walk, progress and
  failure mapping move to one stdlib-only package; `local_operator/imagegen/`
  keeps every pinned public name as aliases/wrappers and keeps all image wire
  code (probes, executors, budgets, labels). Video later imports
  `local_operator.artifacts`, never `local_operator.imagegen`.
- **B — generalise in place inside `local_operator/imagegen/`.** Smallest file
  count, but the "standard" interface would live in a package named for one
  artifact kind; the first video import reads `local_operator.imagegen`, and the
  eventual move doubles the churn. Rejected on the naming defect, not on effort.
- **C — wholesale rename `imagegen` → `artifactgen`.** Honest name, but the
  rename touches ~10 test files, the tool, the TUI, the relay and the guide for
  zero functional gain, inside a PR that must also carry provider breadth.
  Rejected: violates "keep pinned names stable where risk is high".

### 1.2 Recommendation

**A.** Reasons: (1) the interface gets a home that is not named for its first
consumer; (2) imagegen's pinned surface survives as thin aliases, so the
regression risk is mechanical, not semantic; (3) the provider-breadth additions
in the same PR become the seam's first proof (four rungs across three
transports) — exactly how the hub side proved its seam with `altprovider`;
(4) the moved code is small: the tokens, the walk loop, the progress helper and
the failure classifier, ≈350 lines, most of it verbatim moves.

Non-goals: no behaviour change (see §13), no video code, no job registry, no new
tool, no tool-schema growth, no new cancel concept, no cost estimates on the
wire in v1.

## 2. Decisions

| # | Decision | Why |
|---|---|---|
| D1 | New stdlib-only package `local_operator/artifacts/`: tokens (`ArtifactKind`, `JobSpec`, `JobAttempt`, `JobOutcome`, `RungAvailability`, `MediaAsset`, `AttemptOutcome`, `CancelSupport`), the rung seam (`CancelHandle`, `RungResult`, `RungSkipped`, `RungSpec`), the walk (`run_job_walk`), progress (`emit_progress`, `progress_details`), failure mapping (`failure_reason_class`, `REASON_CLASSES`). | One home for the standard interface; import-light like `stt/__init__.py:3-8`, `tts/__init__.py:9-16`, `imagegen/__init__.py:3-4` so session-build paths pay nothing heavy. |
| D2 | `local_operator/imagegen/` keeps **every** pinned public name: `ImageRoute`, `ImageAttempt`, `ImageOutcome`, `ImageRouteResolution`, `resolve_image_route`, `run_image_cascade`, `_run_route`, `run_radient/run_fal/run_openai`, `progress_details`, `emit_progress`, `best_effort_cancel`, `CancelHandle`, `RungSkipped`, `RungResult`, `REASON_CLASSES`, the `availability` probes, `IMAGE_RUNG_TIMEOUT_S`/`IMAGE_GENERATION_TIMEOUT_S`, `_make_pause`, `_preferred_route_label`. Data types become aliases (`ImageAttempt = JobAttempt`); exceptions become subclasses (`ImageGenerationCancelled(JobCancelled)`, `ImageGenerationUnavailable(JobUnavailable)`). | ~10 test files import these names and monkeypatch module attribute (e.g. `cascade._run_route`, `test_cascade.py:112`; `cascade._make_pause`, `:277`; `cascade.IMAGE_GENERATION_TIMEOUT_S`, `:260`; `image_tool._preferred_route_label`, `test_image_tool.py:127`). Stability here is what makes the refactor reviewable. |
| D3 | The rung contract is **(await → `RungResult` \| `RungSkipped` \| raise) under the rung budget**; transport is rung-internal. Poll, single request/response and SSE all satisfy it. | The manager's wave needs all three transports; the walk must not learn about any of them. `asyncio.wait_for` + native httpx cancellation already bound every transport (D6, `rungs.py:3-9`). |
| D4 | Cancel stays exactly the existing machinery: `AbortSignal`-aware `pause` raising the kind's cancellation class; loop task cancellation and the tool's bounded best-effort provider cancel (`image_tool.py:456-484`); `cancel_handle` never set for rungs with no provider-side cancel (OpenAI precedent, `rungs.py:1006-1011`). Each rung DECLARES `CancelSupport` (`none` \| `queued_only` \| `signal`, RFC §5.1) in its `RungSpec`; v1 declares it, tests pin it, no branch reads it yet (extension point §7/§12). | No new cancel concept (design §4 frozen); the RFC vocabulary is reused without forking. |
| D5 | Steer = cancel + a new call composed by the caller; **no upstream steer**. Nothing is built. | RFC §5.8: "Steer is not offered … client-composed cancel + resubmit"; harness steering already cancels `interruptible` tools and pairs a synthetic receipt (`loop.py:3944-3967`), and the next user message composes the next submit. |
| D6 | Restart = a new submit (fresh job); **no job registry, no resume of orphans**; optional `lineage` named as the future additive field (not built). | Design D12 frozen; RFC §5.8 restart = new generate with optional `lineage`; the receipt already carries prompt/model/seed for a re-issue (`guide://image-generation`, "Cancelling and restarting"). |
| D7 | Fail-forward across rungs is preserved verbatim, including after an accepted submit (the image design's contract, `imagegen/__init__.py:8-11`). The RFC's "never fail over after an accepted submit" applies at the **rung** level (a rung is never re-attempted once reached; no resubmit-after-accept anywhere) and is documented as a legitimate divergence at the **walk** level (distinct user-owned accounts; no single reservation — §4). | Behaviour must not change in v1; the divergence is real and must be written down, not silently tightened. |
| D8 | Cost: record only what a provider **reports** on `cost_usd` (Radient today, OpenRouter new); add `cost_source` to the generic types (`reported` \| `rate_table` \| `subscription`) so a future estimate can never masquerade as a charge. Static rate tables (Google, xAI, OpenAI-key) stay guide-level documentation with provenance links; not wired to `cost_usd` in v1. OpenAI-subscription records no cash figure and is marked quota-funded in the guide. | The caption renders "Cost $x" from `cost_usd` (`image_tool.py:587`); putting an unlabelled estimate there would misreport a charge. §9 carries the per-provider table. |
| D9 | The tool description loses its provider enumeration: `"Generate or edit an image via the configured provider; the result is attached for the user. Docs: \`tool://generate_image\`; playbook: \`guide://image-generation\`."` — 160 chars, first sentence 91 (was 186/117), pins updated; providers live in the guide, which costs no schema. | Four new rungs make the parenthetical wrong, and the requirement forbids enumerating providers in the description. First-sentence shape (roster-quotable, `tests/unit/tools/test_image_tool.py:90-93`) is preserved minus the stale list. |
| D10 | Rungs become data: one `RUNG_ORDER` tuple + one `RUNG_SPECS` table in `imagegen/` (route, label, kinds, capabilities, `CancelSupport`, cost source). Additions go through the §5.3 checklist; the existing three keep their order. New rungs append (recommended: openai-sub, google, xai, openrouter — manager sign-off). | Today a rung lives in 7 code sites; the breadth lane must add four without rediscovering them, and video must add its set the same way. |
| D11 | No new payload keys in v1; `kind: "image"|"video"` is the designated additive field, emitted only when a non-image kind exists (§6, §8). Surface reads stay presence-gated. | "Never break current consumers"; nothing consumes `kind` today, and the four surfaces treat unknown keys as absent by construction (`tui/imagegen.py:1-45`, relay/UI/native module docstrings). |
| D12 | One PR, as the manager specified: abstraction first, then the four rungs on top, same branch. If breadth overflows, split within the branch, abstraction commits first. | Manager note 2026-10-09; each addition is then reviewable against the seam it uses. |

## 3. The interface (Python shapes, not Go)

### 3.1 Tokens — `local_operator/artifacts/__init__.py` (import-light)

```python
ArtifactKind = StrEnum(            # wire spelling; only kinds with a real rung set are emitted
    "image", "video",              # video is DECLARED, not built (see §8)
)

AttemptOutcome = Literal["ok", "failed", "skipped"]        # unchanged from imagegen:62-66

@dataclass(frozen=True)
class RungAvailability:            # unchanged field set from imagegen:69-79
    route: str                     # was ImageRoute; str keeps the generic layer kind-free
    available: bool
    reason: str

@dataclass(frozen=True)
class MediaAsset:                  # unchanged field set from imagegen:110-119
    data: bytes
    content_type: str
    source_url: str
    width: int | None = None
    height: int | None = None
    duration_s: float | None = None

@dataclass(frozen=True)
class JobSpec:
    kind: ArtifactKind
    prompt: str
    count: int = 1                 # emitted as the frozen `num_images` payload key (historical name)
    seed: int | None = None
    model: str | None = None
    # NB: kind-specific params (image_size/strength/source_url; a future video's
    # duration/aspect/resolution) stay in the kind's tool params model and reach
    # rungs through the kind's dispatch closure — the walk never parses them.

@dataclass(frozen=True)
class JobAttempt:                  # ImageAttempt today (imagegen:92-107) + nothing new
    route: str
    outcome: AttemptOutcome
    reason_class: str = ""
    message: str = ""
    status_code: int | None = None

@dataclass(frozen=True)
class JobOutcome:                  # ImageOutcome today (imagegen:122-140) + kind/cost_source
    kind: ArtifactKind
    assets: tuple[MediaAsset, ...]
    route: str
    attempts: tuple[JobAttempt, ...]
    model: str = ""
    prompt: str = ""
    seed: int | None = None
    generation_id: str | None = None
    cost_usd: float | None = None                 # only a REPORTED figure (D8)
    cost_source: Literal["reported", "rate_table", "subscription"] | None = None
```

Image compatibility, all in `local_operator/imagegen/__init__.py`:

```python
ImageAttempt = JobAttempt          # alias, not a subclass — one shape, no isinstance traps
ImageOutcome = JobOutcome          # alias
# ImageRoute / ImageRouteResolution stay image-specific (the resolver's `route`
# stays typed ImageRoute; the walk consumes route STRINGS — StrEnum members are
# str, so dict lookups and `== ImageRoute.RADIENT` assertions keep working).
```

### 3.2 The rung seam — `local_operator/artifacts/rung.py`

```python
class CancelSupport(StrEnum):      # RFC §5.1 vocabulary, reused verbatim
    NONE = "none"; QUEUED_ONLY = "queued_only"; SIGNAL = "signal"

@dataclass(frozen=True)
class RungSpec:                    # declared, once, per route
    route: str                     # wire spelling: "radient", "fal", "openai", "openai-sub",
                                   # "google", "xai", "openrouter", ...
    label: str                     # "Radient", "FAL", ... (progress lines, approval text)
    kinds: frozenset[str]          # {"image"} today
    capabilities: frozenset[str]   # RFC §5.2 vocabulary: t2i | i2i | edit | t2v | i2v
    cancel_support: CancelSupport
    cost: str | None               # "reported" | "rate_table" | "subscription" | None

@dataclass
class CancelHandle:                # moved shape from imagegen/rungs.py:241-271, provider: str
    provider: str | None = None
    request_id: str | None = None
    model: str | None = None
    base_url: str | None = None    # poll-rung bases kept so cancel needs no re-resolution
    cancel_url: str | None = None
    credential: SecretStr | None = None
    def clear(self) -> None: ...   # unchanged semantics: filled at accept, cleared at terminal

class RungSkipped(Exception):      # moved verbatim (imagegen/rungs.py:227-238)
    def __init__(self, message: str, *, reason_class: str) -> None: ...

@dataclass(frozen=True)
class RungResult:                  # moved from imagegen/rungs.py:274-282 + cost_source
    assets: list[MediaAsset]
    model: str
    generation_id: str | None = None
    cost_usd: float | None = None
    cost_source: Literal["reported", "rate_table", "subscription"] | None = None
```

**The rung contract, stated once and binding for all three transports:**

1. A rung is `async def call(route) -> RungResult` (a closure the kind builds
   over its concrete params — see §3.3). It either returns assets, raises
   `RungSkipped` (reached but not spent — capability mismatch, affordability),
   or raises any other exception (transport, refusal, timeout → the walk
   classifies and fails forward).
2. It calls `emit(text, details)` for progress, with the canonical payload only
   — stages `queued`/`in_progress` where true; no vendor word escapes.
3. It fills the shared `CancelHandle` the instant a provider job exists and
   `clear()`s it before downloads (existing rule, `rungs.py:763-766`); rungs
   with no provider-side cancel never touch it.
4. It resolves its own credential at call time (existing rule,
   `cascade.py:202-233`) and does not emit cost the provider did not report
   (D8).

Transport shapes the wave must hold, per rung (implementation-internal):

| transport | rungs | shape |
|---|---|---|
| queue poll | Radient, FAL | submit → poll `status` → fetch `result` → bounded downloads (unchanged) |
| single request/response | OpenAI-key, xAI, OpenRouter, Google | one POST under the rung budget; decode b64/url items; downloads as today |
| SSE stream | OpenAI-subscription (Codex backend) | POST `chatgpt.com/backend-api/codex/responses` with `tools:[{"type":"image_generation"}]`, read the SSE stream to completion under the rung budget, emit `in_progress` frames as stream events arrive, decode the image payload from the final event; no seed (manager research; URL constant precedent `providers/clients.py:2374`) |

### 3.3 The walk — `local_operator/artifacts/walk.py`

```python
async def run_job_walk(
    *,
    kind: ArtifactKind,
    spec: JobSpec,
    candidates: Sequence[str],              # available routes, frozen order (§5.3)
    labels: Mapping[str, str],              # route -> label (progress + failure lines)
    call: Callable[[str], Awaitable[RungResult]],   # one rung dispatch (kind's closure)
    handle: CancelHandle,
    emit: ProgressFn | None,
    pause: PauseFn | None,
    rung_timeout_s: float,
    overall_timeout_s: float,
    on_exhausted: Callable[[str, tuple[JobAttempt, ...]], BaseException],
) -> JobOutcome
```

Semantics — byte-for-byte what `run_image_cascade` does today (`cascade.py:351-488`):

- per rung: `remaining = deadline - now`; `remaining <= 0` → `skipped`/`timeout`
  attempt with the exact current sentence ("The overall generation budget was
  spent before this provider was reached."); else `budget = min(rung_timeout_s,
  remaining)` and `await asyncio.wait_for(call(route), timeout=budget)`.
- `JobCancelled` / `asyncio.CancelledError` → **re-raised** (stop, no failover).
- `TimeoutError` → failed attempt, `reason_class="timeout"`, exact current
  sentence ("{label} exceeded its {int(budget)}s generation budget."), mid-walk
  failure update, continue.
- `RungSkipped` → skipped attempt (`exc.reason_class`), continue (no failure
  update — matches today).
- any other exception → `failure_reason_class(exc)` via
  `artifacts/errors.py` (moved from `imagegen/errors.py:82-111`, same
  precedence); failed attempt + mid-walk failure update; continue.
- success → `ok` attempt and `JobOutcome(kind, assets, route, attempts, model=res.model,
  prompt=spec.prompt, seed=spec.seed, generation_id=res.generation_id,
  cost_usd=res.cost_usd, cost_source=res.cost_source)`.
- all failed → `raise on_exhausted(message, tuple(attempts))`; `message` from the
  generic formatter ("{Kind} generation failed on every available provider:" +
  per-attempt lines — for `image` the string is **identical** to today's
  `_all_failed_message`, `cascade.py:309-317`).
- budget constants are PARAMETERS; the walk itself holds no numbers.

`make_pause(signal, cancelled_cls)` also moves here (the current closure,
`cascade.py:176-199`, reads nothing image-specific except which exception it
raises); `cascade._make_pause` becomes a 2-line wrapper binding
`ImageGenerationCancelled` so its pinned call sites (`test_cascade.py:277,290`)
keep working.

### 3.4 The job operations, mapped to the interface

| operation | harness realisation | where |
|---|---|---|
| **submit** | a `generate_image` call → `run_image_cascade` → `run_job_walk`; per rung, the adapter's own submit (POST generate / queue submit / Codex POST) | `image_tool.py:411-513`, walk |
| **progress** | `emit(text, progress_details(...))` — one canonical payload, live-only (design §4.4) | `rungs.py:376-441` |
| **cancel** | the `AbortSignal` pause raises the kind's cancellation class (`JobCancelled` base); the loop's task cancellation propagates after the tool's bounded (5 s) best-effort provider cancel; second Esc abandons the cleanup | `rungs.py:1127-1219`, `image_tool.py:456-484`, `loop.py:3944-3967` |
| **steer** | no upstream operation: steering cancels the wait (tool is `interruptible`, `image_tool.py:166`), the loop pairs its synthetic receipt, the next user message composes the next submit | `loop.py:3944-3967`, `SKIPPED_RESULT_TEXT` (`loop.py:156`) |
| **restart** | a new `generate_image` call — new submit, fresh provider job; nothing resumed; receipt carries prompt/model/seed for re-issue; optional `lineage` is a named future field on `JobOutcome`/details (§7) | `image-generation.md` §4/D12, guide |

## 4. RFC §5 mapping (concept → harness concept, divergences named)

| RFC (MEDIA-PROVIDERS.md) | harness concept | notes / legitimate divergence |
|---|---|---|
| `Provider` adapter (§5.1, spike `provider.go:149-162`) | **rung executor** (`run_radient`/`run_fal`/`run_openai`/new `run_*`, one per route) registered in `RUNG_SPECS` | The harness word is "rung" (STT/TTS/doc precedent); adapters stay plain async functions — the seam between walk and rung is the §3.2 contract, not a class. |
| `Handle` / `Ref` (§5.1, §5.3): stored refs, status/result/cancel addressed by them | `CancelHandle` (`rungs.py:241-271`) + `RungResult.generation_id`; poll URLs carried per-rung | Harness keeps no persisted rows and mints no ids: `tool_call_id` is the caller identity; provider ids (`request_id`, FAL URLs) are recorded for receipts/support. The RFC's id-collision/minting rules are a hub-side concern (its rows are the source of truth there) — N/A here. |
| `State`: Queued/Running/Completed/Failed/Cancelled (§5.5) | progress `stage`: `queued` / `in_progress` / `completed` / `cancelled` / `cancelling`; failure rides `error`/`error_type` with `stage=None` | The wire vocabulary is FROZEN (PR #2089) and every surface already maps it. RFC's Failed-as-terminal-state ≈ harness "settle carries the pair". Unknown provider state must not be passed through — harness maps hub `error_type` vocabulary (`rungs.py:469-471`) and unknown → `upstream` (`rungs.py:705-713`). |
| `Status` fields: queue position, optional progress, logs, metrics (§5.5) | payload `queue_position`, `progress_fraction` (never invented), `log_lines` (verbatim), no metrics key | Metrics beyond `inference_time` are an accepted hub-side passthrough; the harness never renders them, so it does not carry them. |
| `Failure` / `ErrorClass` rejected/rate_limited/unavailable/auth/not_found/indeterminate (§5.1, spike `provider.go:105-121`) | closed `REASON_CLASSES` (`errors.py:29-53`): `refused`, `rate_limited`, `upstream`, `network`, `timeout`, `unauthorized`, `not_found`, `insufficient_*`, `cancelled`, `invalid_response`, `unsupported`, hub `media_*`, `unknown` | Same idea, harness granularity (transport vs status semantics are separated so consumers never parse prose). No `Indeterminate` class today — see the failover row: the harness's post-send timeout/network failures ARE its indeterminate cases, flagged by `timeout`/`network`. |
| Failover: only `RateLimited` / `Unavailable`-NotSent; never `Rejected`/`Auth`/`Indeterminate` after an accepted submit (§5.6) | **Applied at rung level**: a rung is attempted AT MOST ONCE per call; no same-rung resubmit after accept; no resume of a job on another provider. **Divergence at walk level**: fail-forward continues on every class, including failures after an accepted submit — by design (`imagegen/__init__.py:8-11`) | Why divergence is correct here: each rung bills a DIFFERENT account the user owns (hub subscription vs own keys); there is no single reservation whose exactly-once settlement double-billing would threaten; the walk's contract is "get the artifact from any reachable account". The RFC posture still binds wherever one billing boundary exists (hub) and the doc records the residual: a timed-out rung's job may still complete and bill that account (same accepted class as D7 in `image-generation.md`). |
| `CancelOutcome` requested / already_completed / not_found (§5.1) | `best_effort_cancel` tokens: `cancelled` / `already_completed` / `not_found` (+ `none`, `timeout`, `failed`, `abandoned`) (`rungs.py:1127-1219`) | `requested` renders as `cancelled` — matching the media lane's own clause: CANCELLED means *acknowledged, not a promise the job stopped* (RFC §5.8 first bullet; the receipt text already says exactly that, `image_tool.py:249-260`). The extra tokens are harness honesty about its own cleanup (bounded call, double-Esc abandon). |
| `CancelSupport`: none / queued_only / signal (§5.1) | `RungSpec.cancel_support` per route (declared, test-pinned; see §7) | v1 values: radient=signal, fal=signal (its cancel URL answers `CANCELLATION_REQUESTED`; `queued_only` vs `signal` for FAL unverified — research lane), openai-key=none (`rungs.py:1006-1011`), openai-sub=none, google=none, xai=none, openrouter=none (all sync). |
| Restart = new generate, optional `lineage`; steer = client-composed cancel+resubmit (§5.8) | D5/D6 above — identical semantics | `lineage` is the named additive field (§7); nothing to build now. |
| Billing invariants (§5.7): reservation, one-way claim, settlement-on-output, provider cost as row-only metadata | no equivalent layer — the harness never reserves or settles; cost posture is *reporting only* (§9): `cost_usd` on the outcome/details + caption, `cost_source` on the types | Legitimate difference: the hub is a billing system (its invariants protect a ledger); the harness spends the user's own accounts and reports what a provider said. The S-4 idea (upstream cost is metadata, never the charge) survives as: never render a figure no provider reported (D8). |
| Phases, each shipping alone (§5.10) | one harness PR (D12), structured as ordered steps (§10); later rung additions ride the same checklist | The hub phases exist because its live wire has frozen-alias promises; the harness wire is a live-progress stream with no frozen-alias surface, so its first phase can carry the seam + four rungs. |

## 5. The cascade and rungs refactor (what moves, what stays)

### 5.1 What stays exactly where it is

- `local_operator/imagegen/availability.py` — probes only gain new functions;
  the Radient rule (persisted rows only, `:90-105`) and the FAL/OpenAI rule
  (row or exported key, `:108-160`) are FROZEN behaviour; new probes follow the
  credential class each new rung SPENDS with, per that module's own one-class
  rule (`:21-28`). Specifically: `xai` (key rows + `xai-oauth`/OAuth both store
  under `xai`, `providers/registry.py:567-581`), `google` (`api_key` rows +
  `GOOGLE_AI_STUDIO_API_KEY`, row at `:650-664`), `openrouter`
  (`OPENROUTER_API_KEY`, row `:689`), and for the subscription rung an
  **OAuth-grant probe** on `openai` — the deliberate inverse of the
  `openai-key` rule (`availability.py:42-53`): the grant that is invalid at
  `/v1/images` is exactly the credential the Codex backend rung spends, so its
  probe must read the grant class or the rung would be advertised with a
  credential it cannot use.
- Tool surface: one `generate_image` tool, write tier, `interruptible=True`,
  schema unchanged (no `provider` knob, no new params; `GenerateImageParams`
  `image_tool.py:106-134` untouched).
- Result contract: `ToolResult(content=[caption, *AttachmentContent])`,
  `cache_media(..., kind="image", ...)`, details keys, cancellation receipts
  (`image_tool.py:456-603`) — all unchanged.
- Budgets for the existing image rungs (240/300 s, poll cadence, 32 MiB cap,
  5 s cancel) — named constants stay in their current modules.
- The four surfaces — no changes (see §6).

### 5.2 What moves into `local_operator/artifacts/` (verbatim, then re-exported)

| from | to | image-side compatibility |
|---|---|---|
| `imagegen/__init__.py:62-119` (`AttemptOutcome`, `RungAvailability`, `MediaAsset`) | `artifacts/__init__.py` | re-exported under the same names |
| `imagegen/__init__.py:92-107,122-140` (`ImageAttempt`, `ImageOutcome`) | `artifacts/__init__.py` (`JobAttempt`, `JobOutcome` + `kind`/`cost_source`) | aliases `ImageAttempt = JobAttempt`, `ImageOutcome = JobOutcome`; `ImageOutcome` gains `kind` (all construction sites are internal: `cascade.py:473-482`; tests construct `ImageOutcome` nowhere — verified) |
| `imagegen/rungs.py:227-238,241-271,274-282` (`RungSkipped`, `CancelHandle`, `RungResult`) | `artifacts/rung.py` | re-exported from `imagegen.rungs` (tests + tool import there) |
| `imagegen/rungs.py:376-390` (`emit_progress`) | `artifacts/progress.py` | re-exported verbatim |
| `imagegen/rungs.py:393-441` (`progress_details`) | `artifacts/progress.py`, now parameterised `tool: str` (the `tool_name` payload key keeps its slot) | `imagegen.rungs.progress_details` keeps today's EXACT signature (no `tool` param) as a wrapper binding `tool="generate_image"` — pinned call sites (`tests/unit/tools/test_image_tool.py:498-517`) unchanged |
| `imagegen/errors.py:29-53,82-111` (`REASON_CLASSES`, `failure_reason_class`) | `artifacts/errors.py` | re-exported (`tests/unit/imagegen/test_rungs.py` imports `REASON_CLASSES` from `imagegen.errors`); the response-payload builder `api_error_from_httpx_response` (`errors.py:56-79`) STAYS in imagegen (its body-scrubbing contract is lane-specific per its docstring) |
| `cascade.py:176-199` (`_make_pause`) | `artifacts/walk.py` as `make_pause(signal, cancelled_cls)` | `cascade._make_pause` = wrapper binding `ImageGenerationCancelled`; pinned call sites (`test_cascade.py:277,290`) unchanged |
| `cascade.py:300-348` (`_attempt_message`, `_status_code_of`, `_all_failed_message`, `_emit_rung_failure`) | `artifacts/walk.py` (generic; `_all_failed_message` takes labels and kind) | messages byte-identical for the image kind |
| `cascade.py:351-488`'s loop body | `artifacts/walk.py` `run_job_walk` | `run_image_cascade` keeps its full signature and becomes: resolve → candidates → build `JobSpec` → `run_job_walk(...)` with a `_dispatch` closure that calls the module-global `_run_route(route, **kwargs)` (see seam note below) |

**Two seams that MUST survive the move (tests depend on them):**

1. `_run_route` stays a module-level function in `imagegen/cascade.py`;
   `run_image_cascade` calls it through a closure that looks the name up at call
   time (module-global read), so
   `monkeypatch.setattr(cascade, "_run_route", fake)` (`test_cascade.py:112`)
   still intercepts. The closure converts the walk's route STRING back to the
   `ImageRoute` member before calling `_run_route` (the fakes append the member
   and compare with `== ImageRoute.RADIENT`).
2. `cascade.IMAGE_GENERATION_TIMEOUT_S` / `IMAGE_RUNG_TIMEOUT_S` stay module
   attributes; `run_image_cascade` reads them at call time and passes their
   values into the walk (`monkeypatch.setattr(cascade,
   "IMAGE_GENERATION_TIMEOUT_S", 0.0)` at `test_cascade.py:260` drives the
   budget-skip path).

### 5.3 The resolver, `RUNG_ORDER` and the add-a-rung checklist

`resolve_image_route` keeps its shape and reasons; the ordered routes become a
module constant (`IMAGE_RUNG_ORDER`) that the resolver iterates, and each route
gains a `RungSpec` entry (D10). The existing three routes keep their exact
positions and reason strings; new routes append. The `NONE` reason's remedy
sentence gains the new setup paths (append; existing spellings unchanged).

**Tiering posture (reused from the speech cascades, `stt/cascade.py:1-42`,
`tts/cascade.py:1-60`):** hub rung first — Radient, advertised from persisted
login rows only and never the environment (`availability.py:10-19`) — then
BYO-key rungs; probes answer "a credential exists", not "the call will succeed",
with call-time re-resolution and no availability cache (`availability.py:20-28`,
`cascade.py:202-233`); economics follow TTS's (hub path first, then the user's
own keys; a Radient affordability refusal is a SKIP that fails forward,
`rungs.py:632-640`). The image lane's one documented divergence stays as-is:
FAL/OpenAI accept a stored row **or** an exported key (`availability.py:10-19`).
Every new rung inherits the posture, with one binding rule: its probe reads the
credential class it SPENDS with (the `openai-key` rule, `availability.py:42-53`)
— which is why the subscription rung's probe reads the OAuth grant class, the
inverse of that rule, because the grant IS what the Codex backend spends.

**Every rung addition (image or future video) touches exactly these sites —
this checklist IS the seam's contract:**

1. `imagegen/__init__.py` — `ImageRoute` member (wire spelling).
2. `imagegen/availability.py` — one sync, socket-free probe for the credential
   class the rung spends (docstring rule at `:21-28`).
3. `imagegen/cascade.py` — `IMAGE_RUNG_ORDER` entry + `resolve_image_route`
   rung row + reason string.
4. `imagegen/rungs.py` (or a new `imagegen/rungs_<provider>.py` re-exported
   there) — the executor implementing the §3.2 contract for its transport.
5. `_run_route` dispatch arm; `RUNG_SPECS` entry (`label`, `kinds`,
   `capabilities`, `cancel_support`, `cost`).
6. Cancel path: extend `best_effort_cancel` (`rungs.py:1127-1219`) with the
   provider's arm, or (if none) leave the handle unset — the OpenAI precedent.
7. Tests: resolver matrix row; wire shape against `httpx.MockTransport`; the
   capability skip (if any) as `RungSkipped`; cancel (if any); progress
   payload conformance.
8. Docs: `guide://image-generation` provider table + setup + cost notes;
   `login_catalog` prose ONLY if a new loginable row appeared (all four wave
   rungs reuse existing rows — `providers/registry.py` and
   `login_catalog.py:85-92,151-158`).
9. Approval copy: `_preferred_route_label` (`image_tool.py:172-191`) gains the
   label (user-visible text; pinned test updates).

Order recommendation for the wave's four (manager sign-off; one constant):
`openai-sub` (quota-funded, no marginal cash) → `google` → `xai` → `openrouter`.
Rationale: subscription spend is already paid; the rest follow the research
lane's order. NOT changing: the relative order of Radient → FAL → OpenAI-key.

## 6. The progress/UX contract for the four surfaces

The canonical payload is emitted by ONE function (`progress_details`,
`rungs.py:393-441`) and read by four surfaces through one adapter each. v1
changes NOTHING on this wire; the table below is the contract the refactor must
preserve, plus the named additive extensions.

| payload key | producer today | TUI (`tui/imagegen.py`) | relay (`mobile/web/src/lib/image-gen.ts`) | desktop (`image-gen-card-model.ts` @ UI origin/main) | native (`imagegen.ts` @ mobile origin/main) |
|---|---|---|---|---|---|
| `stage` (`queued`/`in_progress`/`completed`/`cancelled`/`cancelling`/`None`) | rungs + tool cancel stages (`image_tool.py:296-316,456-483`) | `live_from_details` | `imageGenView` (unknown word = absent) | `PHASE_OF_STAGE` | stage-first mapping |
| `queue_position` | poll rungs | yes | yes | yes | yes |
| `progress_fraction` | always `None` (never invented) | indeterminate canvas `:56-60` | presence-gated | presence-gated | presence-gated |
| `log_lines` | Radient/FAL passthrough | verbatim, tail | verbatim | verbatim | verbatim |
| `error` / `error_type` | mid-walk failure (`stage=None`) + cancel conflict (`media_already_completed`) | pair-driven | `finished` arm | `finished` routing | `finished` routing |
| `provider` / `model` / `elapsed_s` / `num_images` | rungs | (context) | (context) | (context) | (context) |
| `tool_name` | `"generate_image"` | detection set `IMAGE_GEN_TOOLS` | `IMAGE_GEN_TOOLS` | tool-name set | `IMAGEGEN_TOOLS` |
| **additive (not emitted in v1)** `kind` | — | future reader | future reader | future reader | future reader |

Rules for future additions (write these into the surface docs with the first
addition):

- **New key** (e.g. `kind`): safe to emit only when at least the four readers
  ignore it gracefully — they do today by construction (presence-gated
  `typeof`/`.get` reads). Still: emit new keys together with the first surface
  wave that consumes them, not before.
- **New stage WORD** is NOT additive: the surfaces treat an unknown word as
  absent (deliberate, e.g. relay docstring "a stranger's vocabulary may not
  repaint the card"). A new word must land in all four surfaces BEFORE it can
  be emitted. Therefore: the stage vocabulary is closed in v1, and video must
  reuse it (or extend it via a programme-wide wave).
- **Non-null new values** for existing keys: same rule as new words when the
  value drives rendering (`error_type` codes); new CODES requiring a branch
  (`media_already_completed` precedent) ship surface-first.

## 7. Cancel / steer / restart semantics (RFC-aligned, nothing new built)

- **Abort / stop**: unchanged machinery (D4). `pause` (a) checks the signal
  before waiting, (b) races `signal.wait()` against the poll interval, (c)
  raises the kind's `JobCancelled` subclass; the tool catches it
  (`image_tool.py:456-475`), emits `cancelling` → `cancelled` stages, runs
  `best_effort_cancel` under the 5 s budget with the double-Esc abandon, and
  returns the clean receipt; loop task cancellation (`CancelledError`) takes the
  same cleanup and re-raises (`:476-484`). The RFC's `CancelOutcome` semantics
  are already implemented as the harness tokens (§4 row).
- **`CancelSupport` per rung (declared, test-pinned; no branch reads it in
  v1)** — see §4 row for values. The named future consumer: the cancel receipt
  could say "this provider cannot be stopped" for `none` rungs (text change →
  needs copy review), and a future surface can render a cancel affordance
  honestly. Until then the silent `none` receipt ("nothing to cancel") — the
  OpenAI precedent — IS the honest behaviour.
- **Steer**: no upstream operation exists or is proposed. The harness's
  steering cancellation + resubmit equals the RFC's "client-composed cancel +
  resubmit"; `lineage` would be optional metadata on the RESUBMIT, not a steer.
- **Restart**: a new submit. Named seam: `JobOutcome.lineage: str | None`
  (unset today) and — IF a future product wants it — a `restart_of` additive
  tool param mapping to `details.lineage = {"restart_of": "<generation_id>"}`.
  RFC §5.8's oracle rule (identical answer for foreign and never-existed ids)
  applies whenever a harness-side store would need to validate it; today there
  is no store and the field would be caller-supplied provenance only. NOT
  built now (tool schema frozen).

## 8. Video extension points (named, not built)

| seam | named shape | what it will need | why it is a seam today |
|---|---|---|---|
| kind | `ArtifactKind.VIDEO` ("video") | a `generate_video`-class tool (sibling of `generate_image`, same write tier/approval pattern) or a kind param — decided by that lane | `ArtifactKind` exists; `kind` is the designated additive payload field (§6) |
| params | per-kind tool params model: `duration_s`, `aspect_ratio`/`resolution`, `seed`, `source_image_path` (i2v) | its own `Params` model + mapping per rung | `JobSpec` deliberately carries only kind-agnostic fields; kind params travel via the dispatch closure (§3.1) |
| budgets | per-kind constants: rung/overall timeout, poll cadence, artifact size cap, download timeout | video numbers are larger (minutes, bigger caps); they are **parameters of `run_job_walk` / the download helper**, never globals inside it | the walk already takes `rung_timeout_s`/`overall_timeout_s`; `download_asset`'s cap is a module constant today (`media.py:25-30`) and becomes a parameter when the first video rung lands |
| capability flags | `RungSpec.kinds` + `RungSpec.capabilities` (`t2i`,`i2i`,`edit`,`t2v`,`i2v` — RFC §5.2 vocabulary) | resolver/dispatch must SKIP a rung that cannot serve the request shape (today lived at runtime: OpenAI img2img skip, `rungs.py:1021-1024`) | declared per rung now; the skip branch it will drive is named, tested for consistency with today's runtime skips |
| assets | `MediaAsset.duration_s` already exists; attachment `kind` derives from MIME major type (`session/attachments.py:258-260`, "video" ready per design D2) | a video result registers with `kind="video"` | the attachment pipeline is already video-ready |
| cost | same §9 posture; video providers will report or need rate tables with provenance | — | `cost_source` vocabulary accommodates both |

Explicitly NOT in this wave: no video tool, no video rung, no video budgets, no
`kind` emission, no video surface work.

## 9. Cost emission points per provider

One rule: **`cost_usd` carries only a figure a provider REPORTED; everything
else is documentation with provenance, and quota-funded calls say so in words,
never in a fabricated number** (D8).

| rung | what the provider returns | emission point today → after | provenance for docs |
|---|---|---|---|
| Radient | `cost_usd` on the generate response (`rungs.py:677`) + unit price in the live models list (`_radient_unit_price`, `:517`) | `RungResult.cost_usd` → `JobOutcome.cost_usd` → `details.cost_usd` + caption "Cost $x." (`image_tool.py:587-592`); set `cost_source="reported"` | reported per call — no table needed |
| OpenRouter (new) | `usage.cost` per request (manager research) | same path, `cost_source="reported"` | reported per call |
| FAL | nothing per request (verified in hub RFC §5.1: "FAL exposes no per-request charge API" — that finding is about the queue API; harness-side same) | `cost_usd` stays `None`; rate table = FAL model pricing listing (per-model) | vendor pricing pages; URL+date to be supplied by the breadth lane (UNVERIFIED here) |
| OpenAI-key | nothing per request (dashboard billing) | `cost_usd` stays `None`; rate table = OpenAI image pricing per model | vendor pricing page; model-dependent (gpt-image/dall-e classes) |
| Google (new) | nothing per request per research | `cost_usd` stays `None`; rate table = Google AI pricing | vendor pricing page; UNVERIFIED here |
| xAI (new) | nothing per request per research | `cost_usd` stays `None`; rate table = xAI pricing | vendor pricing page; UNVERIFIED here |
| OpenAI-subscription (new) | quota-funded; no cash figure (research) | `cost_usd` stays `None`; guide states quota-funded; `cost_source="subscription"` reserved on the types | none (not a cash spend) |

Ledger: the harness has none — no reservation, no settlement, no usage record
(RFC §5.7 is hub-side; §4 table records the mapping). The user-facing cost
surfaces are the approval prompt (quantity/size/provider pre-spend,
`image_tool.py:213-241`), the caption (post-spend, reported only), and the
guide's cost section, updated in this PR for the new rungs.

## 10. File-by-file migration plan (ordered by risk, one PR)

Order = risk descending; each step names its gate. All steps are additive or
renames-with-aliases; no step changes the wire.

1. **Baseline.** Run the targeted image suites on the base (`imagegen/*`,
   `tools/test_image_tool`, `tui/test_imagegen*`) and record results — the
   refactor's before-picture. (Fast; no heavy gates.)
2. **New package `local_operator/artifacts/`** — `__init__.py`, `rung.py`,
   `progress.py`, `walk.py`, `errors.py` per §3 (no importers yet) + unit tests
   for the generic pieces (`tests/unit/artifacts/`): walk matrix, `make_pause`,
   progress key set, classifier. Gate: new tests + flake8/black/isort/pyright
   on changed files.
3. **Rewire `imagegen` onto it** — the moves/aliases of §5.2, in this order
   inside the step: `errors.py` re-export → `rungs.py` re-exports + wrapper
   `progress_details` → `cascade.py` (`_make_pause` wrapper, `run_job_walk`
   delegation, `_run_route` closure seam, budget read-through). Gate: ALL of
   `tests/unit/imagegen/`, `tests/unit/tools/test_image_tool.py`,
   `tests/unit/tui/test_imagegen*.py` unchanged and green. This step must not
   touch test files.
4. **Tool text** — description rewrite (D9) + `_preferred_route_label` labels
   for new routes + the two pinned tests updated (`len == 160`, first == 91;
   none of the assertion STRUCTURE changes). Gate: `test_image_tool.py`; eyeball
   the diff (model-facing text).
5. **Rung data** — `IMAGE_RUNG_ORDER` + `RUNG_SPECS` (+ per-route
   `cancel_support`/`cost`/`capabilities` entries for the existing three, values
   as §4), resolver iterates the constant. Gate: `test_cascade.py` green with
   no edits.
6. **Breadth rung 1: `openai-sub`** (SSE transport — the one new transport):
   probe (OAuth-grant class), executor, dispatch arm, spec entry, cancel=None,
   cost=subscription, tests (MockTransport SSE script), guide row. Gate:
   targeted rung tests + resolver matrix.
7. **Breadth rung 2: `google`** (single request; new param mapping `response_format`);
   3: **`xai`** (OpenAI-images-shaped; key OR OAuth); 4: **`openrouter`**
   (reported cost). Each rides the checklist (§5.3) + tests. Gates per rung.
8. **Docs** — guide provider table/setup/cost; `docs/design/image-generation.md`
   pointer to this doc if the manager wants; TOOL_NOTES only if text changed.
9. **Terminal gate** — scoped `make check-changed` during the loop; at the frozen
   head the full-tree gate per repo policy (heavy runs bounded/reaped per
   AGENTS.md; CI authoritative). PR body: testing evidence per the operator's
   standard (real-path run with a live rung if any credential is available —
   else state honestly which rungs were exercised only against MockTransport).

## 11. Test strategy (targeted; full trees only at the terminal gate)

- **Refactor conformance (the big one, zero new tests):** the existing suites
  are the refactor's oracle — `tests/unit/imagegen/test_cascade.py` (walk
  semantics through the pinned `_run_route` seam), `test_rungs.py` (wire shapes,
  cancel tokens, progress), `test_media.py`, `test_availability.py`,
  `tests/unit/tools/test_image_tool.py` (tool contract + canonical payload +
  description pins), `tests/unit/tui/test_imagegen*.py` (surface mapping). They
  must pass UNEDITED through steps 2–5 (except the two description pins in
  step 4, which are deliberate).
- **New generic tests** (`tests/unit/artifacts/`): walk matrix with a fake
  `call` (first-wins, per-class fail-forward, skip, budget exhaustion, cancel
  stop, exhausted message + factory), `make_pause` abort race (mirror of
  `test_cascade.py:275-300`), progress key set (all keys present on every
  update), classifier precedence, and ONE seam-proof test:
  `run_job_walk(kind=VIDEO, ...)` with a fake call — proves the walk is
  kind-neutral without building video.
- **Alias identity tests** (cheap, in `tests/unit/imagegen/`): `ImageAttempt is
  JobAttempt`, `ImageOutcome is JobOutcome`; plus the monkeypatch seams still
  intercept (`_run_route`, `_make_pause`, budget constant) — implicitly covered
  by existing tests, explicitly by one new test that patches each name.
- **Wire conformance on all four surfaces:** the two in-repo consumers keep
  their existing tests green (TUI `test_imagegen.py` — stage map + canonical
  nulls; relay `image-gen.test.ts` — vitest in `local_operator/mobile/web`);
  the two sibling-repo consumers are NOT touched by this PR and their suites
  (`~/local-operator-ui` `scripts/image-gen-card.test.mjs`; `~/local-operator-mobile`
  `src/features/session/imagegen.test.ts`) run in their own repos if/when a
  future wave touches the payload. Add ONE harness-side test asserting
  `progress_details()` emits the full canonical key set and the frozen stage
  vocabulary, so any future field addition fails loudly here first.
- **Breadth rung tests** (per rung, following `test_rungs.py` patterns):
  `httpx.MockTransport` wire shape; resolver row; fail-forward per class;
  cancel arm or its documented absence; cost emission (reported only);
  approval label; progress stage conformance.
- **Not in scope here:** full unit tree / `make type-check` (heavy; terminal
  gate or CI), surface visual waves (no visual change), e2e.

## 12. Risks and rollback

| # | risk | mitigation |
|---|---|---|
| R1 | The walk extraction changes a message byte or attempt shape the model reads | messages/attempts pinned by existing tests; step 3 gate requires them green unedited; diff review focuses on the four exact strings |
| R2 | Alias/subclass drift (`ImageAttempt` vs `JobAttempt` diverging later) | aliases (same object) + identity tests; types live in ONE module |
| R3 | A pinned monkeypatch seam breaks (`_run_route`, budget constant, `_preferred_route_label`) | all listed in §5.2/§2 D2 with their test citations; covered by running the unedited suites |
| R4 | New rungs quietly change behaviour for existing users (e.g. a resolver reorder, or a new probe answering for the wrong credential class) | append-only order; probes must match the spent credential class (`availability.py:21-28` rule); resolver matrix tests pin the first-match answers per credential set |
| R5 | SSE rung (openai-sub) behaves differently under abort (stream continues provider-side) | it declares `CancelSupport.none` exactly like the OpenAI-key precedent; the receipt and guide say "may still complete" — the honest existing wording covers it |
| R6 | Cost misreporting via future rate tables | `cost_source` on the types; documented rule (D8); caption renders only reported figures |
| R7 | Description rewrite churns the model's tool knowledge | minimal diff (drop the stale list only), first-sentence shape preserved, pins updated deliberately in step 4 |

Rollback: the whole change is code-internal — no wire migration, no config, no
persistent state, no data. Reverting the PR restores the previous tree exactly;
per-rung rollback is removing one entry from `IMAGE_RUNG_ORDER`/`RUNG_SPECS`
(+ its probe), which simply returns the walk to the previous rung set. A bad
new rung can also be neutralized by removing its credential (availability rule)
— though that is user-visible, so prefer the code revert. Ship strategy: one PR
on `feat/media-provider-breadth`; if the manager splits, the abstraction
(commits 2–5) lands first and the rungs follow on the same branch.

## 13. Behaviour-identical statement (v1)

For every existing user, on the image path, v1 changes **no runtime behaviour**:

- identical progress payload (keys, values, nulls, stage words, strings);
- identical rung order, budgets, poll cadence, download caps, cancel budgets;
- identical attempts/resolution records, error-note sentences, cancel receipts;
- identical tool schema, approval text for existing providers, result/caption
  shape, attachment registration;
- identical availability answers for the existing three routes (new probes are
  additive functions; the gate's answer changes ONLY when a new credential
  exists — which is the wave's point);
- no new wire keys, no new stages, no config keys, no persistent state.

The deliberate, reviewable deltas (text-only, model-facing): the tool
description loses its stale provider list (D9, pins updated in-PR), the
availability `NONE` remedy sentence appends the new setup paths, and the guide
gains the new provider rows. Nothing a surface renders changes.

## 14. Open questions (for the manager / research lane)

1. **Rung order and membership sign-off** — recommendation in §5.3
   (`openai-sub, google, xai, openrouter`, appended after the frozen three).
2. **FAL `CancelSupport` value** — `signal` vs `queued_only` (its cancel URL
   answers `CANCELLATION_REQUESTED`; whether mid-run cancel is honoured is
   unverified). Affects the spec value only.
3. **Rate tables** — docs-only with provenance (recommended, D8) vs wiring
   labelled estimates into details later. Needs vendor pricing URLs+dates from
   the research lane either way.
4. **Description rewrite** — accept D9's 160/91-char wording?
5. **OpenAI-sub vs OpenAI-key precedence** when both credentials exist
   (subscription first — no marginal cash — vs key first — seed support,
   `rungs.py:1017-1019`). Recommendation: subscription first; document the
   seed trade-off in the guide.
6. **`lineage`** — keep as a named seam (recommended) or build the additive
   details field now (no consumer yet).
7. **`kind` emission** — with video only (recommended) or emit `"image"` now
   for four-surface reader rehearsal.

---

*Derived from the sources cited inline; all file:line references are against
`c5dee91c9c` (harness), `origin/main` of `~/local-operator-ui` and
`~/local-operator-mobile` for their two readers, and the merged RFC + spike
branch for the hub vocabulary. Unverified items are labelled (FAL cancel
semantics; rate-table provenance; the new rungs' wire details rest on the
manager's research note — the code-level design treats them as adapters on the
seam, which is what this document must hold regardless of their final wire
shapes).*
