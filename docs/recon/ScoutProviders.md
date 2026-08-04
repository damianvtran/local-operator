# omp provider / auth layer — recon report

Repo: `~/oss/oh-my-pi` (Bun + TypeScript monorepo). Paths are relative to the repo root.
`packages/ai` (`@oh-my-pi/pi-ai`) owns providers/auth/streaming; `packages/catalog` (`@oh-my-pi/pi-catalog`)
owns the model catalog; `packages/coding-agent` owns session-level model failover.

> **Delivery note:** `local://scout-omp-providers.md` could not be created. This subagent has no `write`
> tool, and the `node_repl` MCP sandbox returns `EPERM` for every path tried (cwd `/Users/damian/tmp`,
> `~/.omp`, the session `local/` directory, and the OS temp dir). This field IS the report — persist it
> from `agent://ScoutProviders` if a file is required.

---

## 1. Provider registry — declaration & discovery

### 1.1 Two orthogonal registries (do not confuse them)

| Registry | File | Keyed by | Answers |
|---|---|---|---|
| **Provider/auth registry** | `packages/ai/src/registry/registry.ts` | provider id (`anthropic`) | whose credentials |
| **Custom API registry** | `packages/ai/src/api-registry.ts` | API id (`anthropic-messages`) | which HTTP wire shape |
| Model catalog | `packages/catalog/src/` | provider id → models | which models, baseUrl, ctx |

One provider can use several apis; one api serves dozens of providers.

### 1.2 `ProviderDefinition` — the single per-provider record

`packages/ai/src/registry/types.ts`. This is the *whole* auth model; the `OAuthProvider` union,
`serviceProviderMap`, the login list, refresh dispatch, and CLI callback-port maps are all **derived**.

```ts
export interface ProviderDefinition {
  readonly id: string;                 // "anthropic"
  readonly name: string;               // "Anthropic (Claude Pro/Max)"
  readonly available?: boolean;        // default true
  readonly showInLoginList?: boolean;  // default true when `login` present

  // API-key path
  readonly envKeys?: KeyResolver;            // env var name OR () => string|undefined
  readonly allowsMissingApiKey?: boolean;    // transport self-authenticates (Bedrock/Vertex ADC)

  // provider-owned request shaping
  readonly prepareRequest?: ProviderRequestPreparer;
  readonly mapSimpleOptions?: ProviderSimpleOptionsMapper;
  readonly prepareModelDiscovery?: ProviderModelDiscoveryPreparer;

  // interactive login
  readonly login?: (cb: OAuthLoginCallbacks) => Promise<OAuthCredentials | string>;
  readonly refreshToken?: (c: OAuthCredentials) => Promise<OAuthCredentials>;
  readonly getApiKey?: (c: OAuthCredentials) => string;
  readonly storeCredentialsAs?: string;      // openai-codex-device => openai-codex

  // login UX
  readonly callbackPort?: number;      // presence => entry in broker CALLBACK_PORTS
  readonly pasteCodeFlow?: boolean;    // => member of PASTE_CODE_LOGIN_PROVIDERS
}
export type KeyResolver = string | (() => string | undefined);
export const AUTHENTICATED_SENTINEL = "<authenticated>";
```

**Field presence is the feature flag.** Heavy OAuth modules are reached through **dynamic-import thunks**
so they stay out of the eager startup graph:

```ts
// packages/ai/src/registry/anthropic.ts
envKeys: () => isFoundryEnabled()
  ? $pickenv("ANTHROPIC_FOUNDRY_API_KEY","ANTHROPIC_OAUTH_TOKEN","ANTHROPIC_API_KEY")
  : $pickenv("ANTHROPIC_OAUTH_TOKEN","ANTHROPIC_API_KEY"),
login: async (cb) => (await import("./oauth/anthropic")).loginAnthropic(cb),
refreshToken: async (c) => (await import("./oauth/anthropic")).refreshAnthropicToken(c.refresh),
callbackPort: 54545,
pasteCodeFlow: true,
```

### 1.3 Registration & derivation

`registry.ts` holds one flat `const ALL = [...]` of ~76 definitions. Adding a provider = one new file
in `registry/` + one line in that array.

```ts
export const PROVIDER_REGISTRY: readonly ProviderDefinition[] = ALL;
const BY_ID = new Map(ALL.map(p => [p.id, p]));
export function getProviderDefinition(id: string): ProviderDefinition | undefined;
export type OAuthProviderUnion = Extract<RegistryDef, { login: object }>["id"];

// compile-time completeness gate against the model catalog:
type _MissingCatalogProviders = Exclude<KnownProvider, RegistryDef["id"]>;
true satisfies _MissingCatalogProviders extends never ? true : [...];
```

Derived: `registry/derived.ts` → `PASTE_CODE_LOGIN_PROVIDERS`; `registry/oauth/index.ts` →
`builtInOAuthProviders` (filter `login && showInLoginList !== false`).

### 1.4 Custom/extension API registration (`api-registry.ts`)

Reserved built-in `KnownApi` ids (cannot be shadowed):

```
openai-completions · openai-responses · openrouter · openai-codex-responses
azure-openai-responses · anthropic-messages · bedrock-converse-stream
google-generative-ai · google-gemini-cli · google-vertex · ollama-chat
cursor-agent · gitlab-duo-agent · devin-agent
```

```ts
registerCustomApi(api, streamSimple, sourceId?, stream?)   // throws on reserved name
getCustomApi(api); unregisterCustomApis(sourceId); clearCustomApis();
```
Same pattern for OAuth: `registerOAuthProvider / getOAuthProvider / unregisterOAuthProviders`.

### 1.5 Complete provider list (all 76, registry order)

```
azure                    openai-codex             anthropic                zai
zai-coding-plan          kimi-code                openrouter               github-copilot
cursor                   devin                    google-antigravity       google-gemini-cli
openai-codex-device      xai                      xai-oauth                gitlab-duo
gitlab-duo-workflow      alibaba-coding-plan      alibaba-token-plan       aiand
aimlapi                  zhipu-coding-plan        umans                    qwen-portal
sakana                   minimax-code             minimax-code-cn          xiaomi
xiaomi-token-plan-sgp    xiaomi-token-plan-ams    xiaomi-token-plan-cn     firepass
deepseek                 meta                     moonshot                 cerebras
baseten                  fireworks                together                 nvidia
novita                   huggingface              perplexity               qianfan
venice                   siliconflow              siliconflow-cn           synthetic
nanogpt                  wafer-serverless         coreweave                vercel-ai-gateway
cloudflare-ai-gateway    litellm                  kilo                     zenmux
opencode-zen             opencode-go              tavily                   kagi
exa                      parallel                 ollama                   ollama-cloud
lm-studio                llama-cpp                vllm                     openai
google                   google-vertex            groq                     mistral
minimax                  amazon-bedrock           bedrock-mantle           gmi-cloud
```

- `tavily / kagi / exa / parallel` are **search/tool credentials**, deliberately unified into the same registry.
- `ollama / lm-studio / llama-cpp / vllm` are local OpenAI-compatible endpoints.
- `amazon-bedrock / bedrock-mantle / google-vertex` use `allowsMissingApiKey` + credential chains
  (`providers/aws-credentials.ts`, `utils/aws-profile.ts`).

### 1.6 `provider-details.ts` — display only

Not part of resolution. Builds `ProviderDetails{provider, api, fields:{label,value}[]}` for the UI status
panel (Model / API / Auth / Endpoint / Source, plus Codex-only Transport / WebSocket / Reuse via
`getOpenAICodexTransportDetails()`). `credentialSource` comes from `AuthStorage.describeCredentialSource()`.

---

## 2. OAuth flows

### 2.1 CRITICAL: real OAuth vs. paste-an-API-key "login"

76 providers have `login`, but **most are not OAuth**. `registry/api-key-login.ts` exports
`createApiKeyLogin(config)` returning `(ctrl) => Promise<string>`: opens a dashboard URL, prompts for a
paste, optionally validates, returns the trimmed key. Its own docstring: *"Several providers … don't
actually implement OAuth — they just ask the user to paste an API key."*

```ts
export type ApiKeyLoginConfig = {
  providerLabel; authUrl; instructions; promptMessage; placeholder;
  validation: ChatCompletionsValidation | AnthropicMessagesValidation
            | ModelsEndpointValidation | null;
};
```
Validators in `registry/api-key-validation.ts`.

**Discriminator when porting:** `login` returning `string` ⇒ API-key paste (stored as
`ApiKeyCredential{source:"login"}`); returning `OAuthCredentials` ⇒ real OAuth. `AuthStorage.login()`
branches on `typeof result === "string"`.

Non-OAuth `createApiKeyLogin` providers: aiand, baseten, cerebras, coreweave, exa, firepass, fireworks,
gmi-cloud, huggingface, kagi, litellm, llama-cpp, lm-studio, meta, minimax-code(+cn), moonshot, nanogpt,
nvidia, ollama(-cloud), parallel, qianfan, qwen-portal, sakana, siliconflow(+cn), synthetic, tavily,
together, umans, venice, vercel-ai-gateway, vllm, wafer-serverless, xai, xiaomi(+3 regions), zai, zenmux,
zhipu-coding-plan, deepseek, cloudflare-ai-gateway, kilo, novita, alibaba-*, opencode-zen/go.

### 2.2 Shared OAuth machinery

**`registry/oauth/types.ts`**
```ts
export type OAuthCredentials = {
  refresh: string; access: string; expires: number;   // epoch ms
  enterpriseUrl?; projectId?; email?; accountId?; apiEndpoint?;
  orgId?;        // org/workspace scope — captured at login, NEVER rewritten on refresh
  orgName?;
  authorizedAt?; // epoch ms of interactive login; refreshes preserve it
};
export type OAuthAuthInfo = { url: string; launchUrl?: string; instructions?: string };
export interface OAuthController {
  onAuth?(info); onProgress?(msg); onManualCodeInput?(): Promise<string>;
  onPrompt?(prompt): Promise<string>; signal?: AbortSignal; fetch?: FetchImpl;
}
```
`orgId` is why one email can hold multiple subscriptions as **separate credential rows**.

**`registry/oauth/pkce.ts`** — 96 random bytes → base64url verifier; `crypto.subtle.digest("SHA-256")` →
base64url challenge (S256).

**`registry/oauth/callback-server.ts`** — abstract `OAuthCallbackFlow`. Subclasses implement only
`generateAuthUrl(state, redirectUri)` and `exchangeToken(code, state, redirectUri)`.

```ts
interface OAuthCallbackFlowOptions {
  preferredPort: number;
  callbackPath?: string;        // default "/callback"
  callbackHostname?: string;    // default "localhost" (binds 127.0.0.1)
  redirectUri?: string;         // exact URI => disables port fallback
  allowPortFallback?: boolean;  // default true
  manualInputOnly?: boolean;    // no server; user pastes code
}
```
Behaviour worth copying verbatim:
- Server starts **before** the auth URL is generated, so the *actual* bound port lands in `redirect_uri`
  (fallback binds port `0` when the preferred port is busy).
- Two routes on one server: `callbackPath` (provider redirect) and **`/launch`** — a 302 to the pending
  auth URL, advertised as `launchUrl` so a TUI can offer a ~30-char copy target that viewport truncation
  cannot corrupt. `#launchUrlIfSafe()` suppresses it for non-http(s), non-loopback, or path-colliding URIs.
- `allowPortFallback:false` throws `ConfigurationError` *before* opening the browser when the provider
  validates redirect URIs (avoids an opaque 500 plus a 5-minute hang).
- `DEFAULT_TIMEOUT = 300_000`. Cancellation via `ctrl.signal` → `LoginCancelledError`. `server.stop()` in `finally`.

**`registry/oauth/device-code.ts`** — RFC 8628 poller.
```ts
type OAuthDeviceCodePollResult<T> =
  | {status:"complete"; value:T} | {status:"pending"} | {status:"slow_down"} | {status:"failed"; message};
pollOAuthDeviceCodeFlow<T>({ poll, intervalSeconds=5, expiresInSeconds, signal }): Promise<T>
```
Min interval 1000 ms; each `slow_down` adds 5000 ms; a dedicated timeout message calls out WSL/VM clock drift.

**`registry/oauth/google-oauth-shared.ts`** — `GoogleOAuthFlow` + `oauthFetch()` (composes caller signal
with a 30 s per-request timeout) + `throwIfLoginCancelled()`.

### 2.3 Per-provider OAuth flows

#### Anthropic (Claude Pro/Max) — `registry/oauth/anthropic.ts`
- **Flow:** authorization code + PKCE (S256), loopback callback. Port `54545`, path `/callback`. `pasteCodeFlow: true`.
- **client_id:** `9d1c250a-e61b-44d9-88ed-5944d1962f5e` (stored base64, `atob`'d).
- **Endpoints:** authorize `https://claude.ai/oauth/authorize`; token `https://api.anthropic.com/v1/oauth/token`;
  identity `https://api.anthropic.com/api/claude_cli/bootstrap?entrypoint=cli&model=…`.
- **Scopes:** `org:create_api_key user:profile user:inference user:sessions:claude_code user:mcp_servers user:file_upload`.
  A code comment warns that `platform.claude.com/oauth/authorize` issues console tokens **without**
  `user:inference` — the `claude.ai` endpoint is required for direct inference.
- Auth params also send `code: "true"`.
- **Exchange:** JSON POST (no `Accept` header — mirrors Claude Code) with
  `grant_type, client_id, code, state, redirect_uri, code_verifier`. A `#` fragment in a pasted code splits
  into `code#state`.
- **Refresh:** JSON POST `grant_type=refresh_token, client_id, refresh_token` plus headers
  `anthropic-beta: oauth-2025-04-20` and `User-Agent: anthropic-sdk-typescript/0.94.0 userOAuthProvider`.
  Refresh **omits org fields on purpose** so merge-over-stored preserves them.
- **Expiry:** `Date.now() + expires_in*1000 - 5min`.
- **Grant TTL:** `ANTHROPIC_OAUTH_GRANT_TTL_MS = 30d` — the whole refresh-token *family* dies 30 days after
  authorization regardless of rotation.
- Identity prefers inline `account{uuid,email_address}` / `organization{uuid,name}`, falling back to
  `/bootstrap`. `includeOrg` is **login-only**.

#### OpenAI Codex (ChatGPT Plus/Pro) — `registry/oauth/openai-codex.ts`
Two flows, two provider ids, one credential row (`storeCredentialsAs: "openai-codex"`).

*Browser* (`openai-codex`): code+PKCE, **port 1455 fixed**, path `/auth/callback`, `redirectUri` pinned to
`http://localhost:1455/auth/callback` (OpenAI allowlist ⇒ no port fallback). client_id
`app_EMoamEEZ73f0CkXaXp7hrann`; authorize `https://auth.openai.com/oauth/authorize`; token
`https://auth.openai.com/oauth/token`; scope
`openid profile email offline_access api.connectors.read api.connectors.invoke`; extra params
`id_token_add_organizations=true`, `codex_cli_simplified_flow=true`, `originator`. Exchange is form-encoded, 15 s timeout.

*Device* (`openai-codex-device`, **not** RFC 8628 — OpenAI-private endpoints):
`POST https://auth.openai.com/api/accounts/deviceauth/usercode {client_id}` → `{device_auth_id, user_code, interval}`;
poll `POST …/deviceauth/token` (max 120 polls, 5 s + 3 s safety margin); user visits
`https://auth.openai.com/codex/device` and types `user_code`; success returns
`{authorization_code, code_verifier}` exchanged against `redirect_uri = https://auth.openai.com/deviceauth/callback`.

**Identity from JWT claims**, no extra API call: `decodeJwt` reads `https://api.openai.com/auth →
{chatgpt_account_id, chatgpt_plan_type}` and `https://api.openai.com/profile → {email}`.
`orgId = accountId`, `orgName = planType`. Login **fails hard** without `accountId`.
**Refresh:** form POST `grant_type=refresh_token, refresh_token, client_id`; org fields omitted.

#### Kimi / Moonshot — `registry/oauth/kimi.ts` (provider `kimi-code`)
- **Flow:** RFC 8628 device authorization grant. **No callback server.**
- client_id `17e5f671-d194-4dfb-9706-5516cb48c098`; host `https://auth.kimi.com`
  (override `KIMI_CODE_OAUTH_HOST` / `KIMI_OAUTH_HOST`).
- `POST /api/oauth/device_authorization` (form, `client_id`) → `{user_code, device_code, verification_uri,
  verification_uri_complete, expires_in, interval}`.
- Poll `POST /api/oauth/token` with `grant_type=urn:ietf:params:oauth:grant-type:device_code`; handles
  `authorization_pending` / `slow_down` (+5 s, honours returned `interval`) / `expired_token` / `access_denied`.
  Defaults: interval 5 s, TTL 15 min, expiry skew 5 min.
- **Device fingerprint headers** on every OAuth *and* usage call (`getKimiCommonHeaders()`):
  `User-Agent: KimiCLI/<ver>`, `X-Msh-Platform: kimi_cli`, `X-Msh-Version`, `X-Msh-Device-Name` (hostname),
  `X-Msh-Device-Model` (os+release+arch), `X-Msh-Os-Version`, `X-Msh-Device-Id`. Device id persisted at
  `<agentDir>/kimi-device-id` mode `0600`, best-effort with a per-process ephemeral UUID fallback.
- **Refresh:** form POST `grant_type=refresh_token`; reuses the old refresh token when the response omits one.
- `moonshot` (plain API) is a **paste-key** provider, distinct from `kimi-code`.

#### xAI / Grok — `registry/oauth/xai-oauth.ts` (provider `xai-oauth`)
- **Flow:** RFC 8628 device code via the shared poller. (Adapted from NousResearch/hermes-agent, MIT.)
- **OIDC discovery:** `GET https://auth.x.ai/.well-known/openid-configuration` → `token_endpoint`, hard-validated
  by `validateXAIEndpoint()` (**https only**, host `x.ai` or `*.x.ai`) — that endpoint receives every future
  refresh token, so it is pinned.
- device code `https://auth.x.ai/oauth2/device/code`; userinfo `https://auth.x.ai/oauth2/userinfo`.
- client_id `b1a00492-073a-47ea-816f-4c329264a828`; scope
  `openid profile email offline_access grok-cli:access api:access`. Poll body
  `grant_type=urn:ietf:params:oauth:grant-type:device_code, client_id, device_code`, `redirect: "error"`.
- Helpers: `parseXAIAccessTokenPayload`, `isXAIAccessTokenExpiring(jwt, skew)`, `extractXAIAccessTokenSubject`,
  `fetchXAIOAuthIdentity`.
- **Separate billing host** (deliberately not `*.x.ai`): `https://cli-chat-proxy.grok.com/v1/billing?format=credits`,
  validated by `validateXAIBillingEndpoint`; header `X-XAI-Token-Auth: xai-grok-cli`.
- `xai` (plain API) is a paste-key provider.

#### GitHub Copilot — `registry/oauth/github-copilot.ts`
- GitHub device flow; client_id `Ov23li8tweQw6odWQebz` (opencode's OAuth app), scope `read:user`.
- `POST https://<domain>/login/device/code` → `POST https://<domain>/login/oauth/access_token`
  (`grant_type=urn:ietf:params:oauth:grant-type:device_code`); `domain` supports GHES/enterprise.
- **No token refresh**: `refreshGitHubCopilotToken` returns `{refresh: t, access: t, expires: now+10y}`.
- Post-login: `discoverGitHubCopilotApiEndpoint` (`GET https://api.github.com/copilot_internal/user` →
  `endpoints.api`) and `enableAllGitHubCopilotModels` — `POST {base}/models/{id}/policy {state:"enabled"}`
  in batches of 5 for every bundled wire model id (Claude/Grok need policy acceptance).
- Poll multipliers 1.2 → 1.4 on `slow_down`.

#### Google Gemini CLI — `registry/oauth/google-gemini-cli.ts`
Authorization code (**confidential client**: client_id *and* client_secret, both base64-embedded), loopback
port **8085**, path `/oauth2callback`, hostname `127.0.0.1`. authorize
`https://accounts.google.com/o/oauth2/v2/auth` (`access_type=offline`, `prompt=consent`); token
`https://oauth2.googleapis.com/token`; scopes `cloud-platform`, `userinfo.email`, `userinfo.profile`.
Post-exchange: userinfo → email; `discoverProject()` against `https://cloudcode-pa.googleapis.com`
(loadCodeAssist / onboardUser LRO polling) → `projectId`. Surfaces a Google *validation-required* message
when `extractGoogleValidationUrl` matches.

#### Google Antigravity — `registry/oauth/google-antigravity.ts`
Same `GoogleOAuthFlow`; port **51121**, path `/oauth-callback`, different client id/secret, extra scopes,
tier `legacy-tier`, project onboarding retries (max 5).

#### Cursor — `registry/oauth/cursor.ts`
**Poll-based, no callback server, no standard token endpoint.** `generateCursorAuthParams()`: PKCE plus a
`crypto.randomUUID()` handshake id → browser at `https://cursor.com/loginDeepControl`; poll
`https://api2.cursor.sh/auth/poll` (max 150 attempts, 1 s base delay); "refresh" via
`https://api2.cursor.sh/auth/exchange_user_api_key`.

#### Devin — `registry/oauth/devin.ts`
Code + PKCE, loopback **59653** `/callback`, `pasteCodeFlow: true`. `https://app.devin.ai` (authorize) /
`https://api.devin.ai/auth/cli/token` (exchange); 1-year fallback expiry.

#### GitLab Duo — `registry/oauth/gitlab-duo.ts`, `gitlab-duo-workflow.ts`
Both code + PKCE, `callbackPort: 8080`, `pasteCodeFlow: true`. **Duo Workflow** advertises a fixed
non-loopback redirect `vscode://gitlab.gitlab-workflow/authentication` (client id
`36f2a70cddeb5a0889d4fd8295c241b7e9848e89cf9e599d0eed2d8e5350fbf5`, scope `api`) — the canonical
`manualInputOnly`/paste case and the reason `#launchUrlIfSafe` rejects custom schemes. `gitlab-duo`
allows a `GITLAB_REDIRECT_URI` env override via `resolveCallbackOptions()`; GitLab rejects unregistered
redirect URIs so port fallback is disabled.

#### Z.AI coding plan — `registry/oauth/zai.ts` (`zai-coding-plan` → stores as `zai`)
Authorization code, loopback **54548** `/callback`, `pasteCodeFlow: true`. authorize
`https://chat.z.ai/api/oauth/authorize`; token `https://zcode.z.ai/api/v1/oauth/token`; then a
**business-login exchange** `https://api.z.ai/api/auth/z/login` minting a durable API key named `oh-my-pi`
(so it never clobbers ZCode's own `zcode-api-key`). Result never expires (`NEVER_EXPIRES = 8.64e15`).
All constants env-overridable (`ZAI_OAUTH_CLIENT_ID`, `ZAI_OAUTH_AUTHORIZE_URL`, `ZAI_OAUTH_TOKEN_URL`,
`ZAI_BIZ_BASE`, `ZAI_BUSINESS_LOGIN_URL`).

#### Perplexity — `registry/oauth/perplexity.ts`
Not a browser OAuth flow: prompt-driven session capture impersonating the macOS app (`API_VERSION 2.18`,
bundle `ai.perplexity.mac`, `Perplexity/641 CFNetwork/1568 Darwin/25.2.0`). JWTs usually omit `exp` ⇒ treated
as non-expiring; special-cased in `getOAuthApiKey`, which re-derives expiry from the JWT `exp - 5min` when present.

#### Summary table

| Provider id | Flow | Port / redirect | Refresh |
|---|---|---|---|
| `anthropic` | code+PKCE | 54545 `/callback` | `refresh_token` (30 d grant cap) |
| `openai-codex` | code+PKCE | **1455 fixed** `/auth/callback` | `refresh_token` |
| `openai-codex-device` | vendor device | none (`auth.openai.com/deviceauth/callback`) | shared with above |
| `kimi-code` | RFC 8628 device | none | `refresh_token` |
| `xai-oauth` | RFC 8628 device + OIDC discovery | none | `refresh_token` |
| `github-copilot` | GitHub device | none | none (10 y static) |
| `google-gemini-cli` | code (confidential) | 8085 `/oauth2callback` | Google `refresh_token` |
| `google-antigravity` | code (confidential) | 51121 `/oauth-callback` | Google `refresh_token` |
| `cursor` | PKCE + poll | none | `exchange_user_api_key` |
| `devin` | code+PKCE | 59653 `/callback` | — |
| `gitlab-duo` | code+PKCE | 8080 `/callback` | yes |
| `gitlab-duo-workflow` | code+PKCE | `vscode://` (paste) | yes |
| `zai-coding-plan` | code → biz key mint | 54548 `/callback` | n/a (never expires) |
| `perplexity` | app-session capture | none | n/a |

---

## 3. auth-storage — persistence

`packages/ai/src/auth-storage.ts` — 8 534 lines. Backend is **SQLite via `bun:sqlite`** (`getAgentDbPath()`).
**No OS keychain.** The broker path (§3.5) is the "vault" story.

### 3.1 Credential model

```ts
export type ApiKeyCredential = { type:"api_key"; key:string; source?:"login" };
export type OAuthCredential   = { type:"oauth" } & OAuthCredentials;
export type AuthCredential    = ApiKeyCredential | OAuthCredential;
export type AuthCredentialEntry = AuthCredential | AuthCredential[];   // N-per-provider
export interface StoredAuthCredential { id:number; provider:string; credential:AuthCredential; disabledCause:string|null }
export type CredentialOriginKind = "runtime"|"config"|"oauth"|"api_key"|"env"|"fallback";
```

### 3.2 SQLite schema

```sql
CREATE TABLE auth_credentials (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  provider TEXT NOT NULL,
  credential_type TEXT NOT NULL,      -- 'api_key' | 'oauth'
  data TEXT NOT NULL,                 -- JSON blob of the credential
  disabled_cause TEXT DEFAULT NULL,   -- soft-delete tombstone
  identity_key TEXT DEFAULT NULL,     -- dedupe key (account/email/project/org)
  created_at INTEGER NOT NULL DEFAULT (<now>),
  updated_at INTEGER NOT NULL DEFAULT (<now>)
);
CREATE INDEX idx_auth_provider ON auth_credentials(provider);
CREATE INDEX idx_auth_provider_identity ON auth_credentials(provider, identity_key) WHERE identity_key IS NOT NULL;

CREATE TABLE auth_credential_blocks (           -- rate-limit / 401 backoff
  credential_id INTEGER NOT NULL,
  provider_key  TEXT NOT NULL,                  -- "${provider}:${credentialType}"
  block_scope   TEXT NOT NULL DEFAULT '',       -- e.g. "tier:fable", "shared", codex meters
  blocked_until_ms INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  PRIMARY KEY (credential_id, provider_key, block_scope)
);
CREATE INDEX idx_auth_credential_blocks_expires ON auth_credential_blocks(blocked_until_ms);

CREATE TABLE auth_credential_refresh_leases (   -- cross-process single-flight
  credential_id INTEGER PRIMARY KEY, owner TEXT NOT NULL, expires_at_ms INTEGER NOT NULL, …);
CREATE TABLE auth_schema_version (id INTEGER PRIMARY KEY CHECK (id=1), version INTEGER NOT NULL);
CREATE TABLE auth_change_revision (id INTEGER PRIMARY KEY CHECK (id=1), revision INTEGER NOT NULL);
CREATE TEMP TABLE auth_local_change_revision (…);   -- distinguishes own writes from peers'
CREATE TABLE auth_credential_block_mirror_guard (credential_id INTEGER PRIMARY KEY) WITHOUT ROWID;
CREATE TABLE cache (key TEXT PRIMARY KEY, value TEXT NOT NULL, expires_at INTEGER NOT NULL);
CREATE TABLE usage_history (id, recorded_at, provider, account_key, limit_id, …);
CREATE TABLE usage_cost_history (id, recorded_at, provider, account_key, cost…);
CREATE TABLE clients (install_id TEXT PRIMARY KEY, hostname, first_seen, last_seen);
CREATE TABLE client_usage (id, recorded_at, install_id, provider, model, tokens…);
```

**Cross-process change detection:** `AFTER INSERT/UPDATE/DELETE` triggers on `auth_credentials` and
`auth_credential_blocks` bump `auth_change_revision`; parallel TEMP triggers bump
`auth_local_change_revision`. `pollExternalChanges()` compares the two, so one omp process notices
another's refresh without touching the provider. PRAGMAs: WAL, `synchronous=NORMAL`, busy timeout.

### 3.3 Selection: round-robin, session stickiness, usage ranking

```ts
#getProviderTypeKey(provider, type) => `${provider}:${type}`
#getNextRoundRobinIndex(providerKey, total)  // (current+1) % total
#getHashedIndex(sessionId, total)            // Bun.hash.xxHash32(sessionId) % total
#getCredentialOrder(providerKey, sessionId, total) // start = hashed(session) else round-robin, then wrap
```
Layered on top:
- **Session stickiness** — `#sessionLastCredential`, cache prefix `session:sticky:`, keeps prompt caches warm.
  Anthropic-only idle window `ANTHROPIC_SESSION_STICKY_CACHE_WARM_MS = 60 min`, after which usage re-ranking
  resumes (Anthropic caps OAuth prompt-cache retention at `ttl:"1h"`).
- **Usage-aware ranking** — per-provider `CredentialRankingStrategy` for `alibaba-token-plan`, `openai-codex`,
  `anthropic`, `google-antigravity`, `zai`. `PRIMARY_WINDOW_HOT_FRACTION = 0.85`: a credential ≥85 % through
  its short (5 h) window is demoted behind cooler siblings.
- **Blocks** — `#markCredentialBlocked`; default backoff `AuthStorage.#defaultBackoffMs = 60 000`, extended by
  the provider's own reported reset time.
- **Codex plan gating** — `resolveOpenAICodexPlanRequirement` / `classifyOpenAICodexPlan` route
  `gpt-5.6-(sol|luna)(-pro)?` to `paid`/`pro` credentials (`free|go` vs `plus|business|team` vs `pro`).

Timing constants: `OAUTH_REFRESH_SKEW_MS = 60 000`, `OAUTH_REFRESH_LEASE_TTL_MS = 15 000`,
`…LEASE_POLL_MS = 50`, `…LEASE_RENEW_MS = 5 000`, `…OPERATION_TIMEOUT_MS = 10 000`,
`USAGE_REPORT_TTL_MS = 5 min`, `USAGE_FAILURE_BACKOFF_MS = 10 000`.

### 3.4 `AuthCredentialStore` — the persistence seam

```ts
export interface AuthCredentialStore {
  close();
  pollExternalChanges?(): boolean;  acknowledgeLocalChanges?();
  listAuthCredentials(provider?): StoredAuthCredential[];
  updateAuthCredential(id, credential);
  deleteAuthCredential(id, disabledCause);
  tryDisableAuthCredentialIfMatches(id, expectedData, …): boolean;
  replaceAuthCredentialsForProvider(provider, credentials): StoredAuthCredential[];
  upsertAuthCredentialForProvider(provider, credential): StoredAuthCredential[];
  deleteAuthCredentialsForProvider(provider, disabledCause);
  getCache(key, {includeExpired}?); setCache(key, value, expiresAtSec);
  deleteCachePrefix?(prefix); cleanExpiredCache();
  getCredentialBlock?(credentialId, providerKey, blockScope): number|undefined;
  recordUsageSnapshots?(…); listUsageHistory?(query); recordClientUsage?(…); getClientUsageSummary?(…);
  // remote (broker) variants: *Remote(...) async twins, markCredentialSuspect?, listDisabledCredentials?
}
```
Implementations: `SqliteAuthCredentialStore` (in-file) and `RemoteAuthCredentialStore`
(`auth-broker/remote-store.ts`, ~52 KB).

### 3.5 Broker mode (remote vault)

`packages/ai/src/auth-broker/` — `server.ts`, `client.ts`, `remote-store.ts`, `discover.ts`,
`snapshot-cache.ts`, `refresher.ts`, `types.ts`, `wire-schemas.ts`.

The broker holds **refresh tokens**; clients get a redacted snapshot:
```ts
export const REMOTE_REFRESH_SENTINEL = "__remote__";
export type RemoteOAuthCredential = Omit<OAuthCredential,"refresh"> & { refresh: "__remote__" };
export type SnapshotCredential = ApiKeyCredential | RemoteOAuthCredential;
```
Wire API (`AUTH_BROKER_API_PREFIX = "/v1"`, default bind `127.0.0.1:8765`):
`GET /v1/healthz` (unauth) · `GET /v1/snapshot` · `GET /v1/snapshot/stream` (SSE: `snapshot`|`entry`|`removed`)
· `POST /v1/credential` · `POST /v1/credential/:id/refresh` · `POST /v1/credential/:id/disable`
· `POST /v1/credential/:id/block` · `DELETE /v1/credential/:id/blocks` · `GET /v1/credentials/disabled`
· `GET /v1/usage` · `GET /v1/usage/history` · `POST /v1/usage/observed` · `GET /v1/usage/clients`
· `POST /v1/usage/stale`.
Defaults: refresh skew 5 min, refresh interval 60 s, snapshot cache TTL 60 min, stream keepalive 20 s,
`DEFAULT_SERVER_IDLE_TIMEOUT_S = 255` (Bun's 10 s default would kill SSE). Capability header
`OMP-Auth-Broker-Capabilities`.

**Discovery precedence** (`discover.ts:resolveAuthBrokerConfig`):
1. `OMP_AUTH_BROKER_URL` / `OMP_AUTH_BROKER_TOKEN`
2. `auth.broker.url` / `auth.broker.token` in `<agentDir>/config.yml|.yaml` (nested or flat dotted key)
3. `<config-root>/auth-broker.token`

`discoverAuthStorage()` returns a broker-backed `AuthStorage` when a URL resolves, else local SQLite.
An encrypted snapshot cache (keyed by token+url) allows offline start; revalidation timeout 500 ms; 401/403
always rethrow. `OMP_AUTH_BROKER_ACCOUNT_POOL_FILE` restricts which identity keys a client may use.

---

## 4. API-key path and oauth-vs-key selection

### 4.1 No user-facing "auth mode" switch — it is a precedence cascade

`AuthStorage.getApiKey(provider, sessionId?, options?)` (`auth-storage.ts:5304`), first match wins:

1. **Runtime override** — `setRuntimeApiKey` (CLI `--api-key`)
2. **Config override** — `setConfigApiKey` (`models.yml` `providers.<name>.apiKey`); documented to beat broker
   OAuth because the user pointed the provider at a custom `baseUrl`/gateway
3. **OAuth credential** — `#resolveOAuthSelection` (auto-refresh, ranking, stickiness)
4. **API key persisted by `/login`** — `ApiKeyCredential` with `source === "login"`
5. **Environment variable** — `getEnvApiKey(provider)`
6. **Stored api_key without `source:"login"`** — e.g. broker-migrated copy, deliberately *after* env
7. **Fallback resolver** — `setFallbackResolver` (custom `models.yml` providers)

Side effect worth porting: before step 5 the session sticky entry is cleared, so `getOAuthAccountId()`
stops emitting `account_uuid` for a request that is no longer OAuth-backed.

`getOAuthAccess()` is the identity-carrying sibling — returns `{accessToken, credentialId, accountId, email,
projectId, enterpriseUrl, apiEndpoint, orgId, orgName}` for providers needing identity headers (Codex
`chatgpt-account-id`, Google `project`, Copilot `enterpriseUrl`). Runtime/config overrides deliberately
short-circuit it to `undefined`.

### 4.2 Env var resolution

`getEnvApiKey / getEnvApiKeyName` live in `packages/ai/src/stream.ts:734`, backed by `serviceProviderMap`
(derived from `ProviderDefinition.envKeys` plus the catalog's `envVars`). `envKeys` may be a computed resolver
(see the Anthropic Foundry example in §1.2).

### 4.3 Structured API keys

`getOAuthApiKey(provider, credentials)` (`registry/oauth/index.ts`) normally returns `creds.access`, but for
**github-copilot, google-gemini-cli, google-antigravity, alibaba-coding-plan** it returns a **JSON string**
`{apiEndpoint, token, enterpriseUrl, projectId, refreshToken, expiresAt, email, accountId}`. It **refuses
expired credentials** (throws) rather than POSTing a `__remote__` sentinel upstream — refresh is exclusively
`AuthStorage`'s job.

### 4.4 Login entry point

`AuthStorage.login(provider, ctrl)` (`auth-storage.ts:2829`):
- synthesises `onManualCodeInput` **only** for `PASTE_CODE_LOGIN_PROVIDERS` — otherwise `OAuthCallbackFlow`
  would race a readline prompt against the HTTP callback and leave a dirty terminal
- looks up `getProviderDefinition(provider) ?? getOAuthProvider(provider)` (built-in then extension)
- `string` result → `ApiKeyCredential{source:"login"}` upsert; `""` → no-op (ollama)
- object result → `OAuthCredential` stamped `authorizedAt: Date.now()`, upserted under
  `def.storeCredentialsAs ?? provider`
- returns `OAuthLoginIdentity {type, email?, accountId?, orgId?, orgName?}`

Health: `checkCredentials(options)` → `CredentialHealthResult[]` with **tri-state** `ok: true|false|null`
(null = no probe configured) plus an optional `CompletionProbe` for a real chat round-trip (a usage endpoint
can 200 while the chat endpoint 401s the same bearer).

---

## 5. Failover / load balancing

Three independent tiers. Do not collapse them.

### Tier 1 — credential rotation inside one provider (`packages/ai/src/auth-retry.ts`)

The **a/b/c policy**:
```ts
export interface ApiKeyResolveContext {
  lastChance: boolean;      // true => rotate to sibling credential
  error: unknown;           // undefined => initial resolve
  previousKey?: string; signal?: AbortSignal;
}
export type ApiKeyResolver = (ctx) => Promise<string|undefined>|string|undefined;
export type ApiKey = string | ApiKeyResolver;
export const AUTH_RETRY_STEPS: readonly boolean[] = [false, true];
export const AUTH_RETRY_MAX_ATTEMPTS = 64;
```
- `error === undefined` → **(a)** initial resolve (cheap, cached token OK)
- `error && !lastChance` → **(b)** force-refresh the *same* account
- `error && lastChance` → **(c)** rotate to a *sibling* credential

`isDirectCredentialRotationError(error)` **skips step (b)** for `isUsageLimit` /
`isInvalidatedOAuthTokenError` / HTTP **403** / `isUsageLimitOutcome` — a valid-but-denied token cannot be
fixed by refreshing, so rotation goes straight through the pool.

```ts
interface AuthRetryKeyState { attemptedKeys:Set<string>; lastKey:string; refreshedCurrent:boolean;
                              legacyAuthSwitchUsed:boolean; attempts:number }
resolveNextAuthRetryKey(state, resolver, error, signal): Promise<string|undefined>
```
Termination: key already in `attemptedKeys`, `attempts >= 64`, resolver returns `undefined`, or signal
aborted. Ordinary 401 gets exactly one refresh-same plus one sibling switch (`legacyAuthSwitchUsed`);
403/usage-limit may cycle every distinct sibling.

Drivers: `withAuth(key, attempt, opts)` (non-streaming: image gen, web search, completions) and
`withOAuthAccess(...)` (hands the attempt the full `OAuthAccess` instead of bare bytes). `stream.ts`
reimplements the same policy with replay-safe buffering. Helpers: `resolveApiKeyOnce`,
`seedApiKeyResolver(seed, resolver)`.

**Resolver construction** — `packages/coding-agent/src/config/api-key-resolver.ts` and
`AuthStorage.resolver(provider, {sessionId, baseUrl, modelId})`:
```ts
if (error === undefined)  return registry.getApiKeyForProvider(provider, sessionId, {baseUrl, modelId});
if (lastChance) {
  const switched = await authStorage.rotateSessionCredential(provider, sessionId, {error, modelId, signal, apiKey: previousKey});
  if (!switched && (isUsageLimit(error) || isUsageLimitOutcome(status, message))) return undefined; // outer layer honours backoff
  return registry.getApiKeyForProvider(provider, sessionId, {baseUrl, modelId});
}
return registry.getApiKeyForProvider(provider, sessionId, {baseUrl, modelId, forceRefresh:true, signal});
```

`AuthStorage.rotateSessionCredential(provider, sessionId, opts)` (`auth-storage.ts:6089`):
- usage-limit / account-rate-limit → `markUsageLimitReached()` (temporary block, sticky preserved)
- otherwise → snapshot sibling availability **before** mutating, clear sticky, block the credential for 60 s;
  if `isInvalidatedOAuthTokenError` **soft-delete the row** (`disabled_cause`) instead; else
  `markCredentialSuspect` (broker) or full `reload()`
- returns `boolean` — "another usable credential of the same type remains"

Provider-specific: `callWithCopilotModelRetry` (`packages/ai/src/utils/retry.ts`) retries Copilot's transient
`model_not_supported` 400 up to 3× (400 ms linear); status-bearing retryables are only re-sent when
`Retry-After` ≤ 30 s.

### Tier 2 — model fallback chains (`packages/coding-agent/src/session/retry-fallback-chains.ts`)

Config surface (`packages/coding-agent/src/config/settings-schema.ts`):

| Setting | Type | Default |
|---|---|---|
| `retry.enabled` | boolean | `true` |
| `retry.maxRetries` | number | `10` |
| `retry.baseDelayMs` | number | `500` |
| `retry.maxDelayMs` | number | `300000` |
| `retry.modelFallback` | boolean | `true` |
| `retry.usageAwareFallback` | boolean | `false` |
| `retry.usageReservePct` | number | `10` |
| `retry.usageReservePolicy` | `confirm`/`auto`/`fail-closed` | `confirm` |
| `retry.fallbackChains` | `Record<string,string[]>` | `{}` |
| `retry.fallbackRevertPolicy` | `cooldown-expiry`/`never` | `cooldown-expiry` |
| `providers.anthropic.serverSideFallback` | boolean | `false` |

```jsonc
{"retry.fallbackChains": {
  "default": ["openai/gpt-4o-mini"],
  "google-antigravity/*": ["google/*", "google-vertex/*"],
  "openrouter/google/*": ["..."]
}}
```
Keys are **roles**, **exact selectors** (`provider/model-id`), or **wildcards** (`provider/*`, id-prefixed
`openrouter/google/*`). A `provider/*` entry keeps the failing model id and swaps provider; an id-prefixed
wildcard re-prefixes the bare id (`google-antigravity/gemini-x` → `openrouter/google/gemini-x`).

`resolveRetryFallbackChainKey(context, currentSelector, currentModel?, roleHint?)` resolves by **specificity**:
exact model key → longest matching wildcard (by id-prefix length) → hinted/configured role → `default`.
`expandDefaultRetryFallbackChains` copies `default` onto every role lacking its own chain.
`validateRetryFallbackChains` reports unknown providers/models as config warnings.
`calculateRetryBackoffDelayMs(base, attempt)` = `min(base·2^(attempt-1), 8000)` with 25 % **downward** jitter.

### Tier 3 — turn recovery (`packages/coding-agent/src/session/turn-recovery.ts`)

Per-turn orchestration (~line 1379):
1. Classify the error; `StaleResponsesItem` → reset the Responses provider session, delay 0.
2. If a recorded usage-limit outcome exists **and** a credential switch already happened (or
   `maybeAutoRedeemCodexReset()` banked a Codex reset credit) → `switchedCredential = true`, delay 0.
3. Else `usageLimitWaitMs = min(provider retry-after, earliest sibling unblock + SIBLING_UNBLOCK_BUFFER_MS)` —
   explicitly so one 60 s sibling block cannot escalate into the provider's multi-hour wait.
4. If no credential switch: `noteRetryFallbackCooldown(selector, retryAfterMs, msg)` then
   `#tryRetryModelFallback(currentSelector, {pinFallback})`, walking `findRetryFallbackCandidates(role, selector)`,
   skipping suppressed selectors, re-clamping to `thinkingLevelCeiling()`. Success ⇒ delay 0.
5. `#tryFireworksFastFallback` — Fast variant → base model, independent of `retry.modelFallback` (intrinsic
   to the Fast contract: speed best-effort, degrade to Standard).
6. Budget exhausted: no fallback ⇒ terminal `auto_retry_end{success:false}`; a fallback ⇒ `#retryAttempt = 1`
   so the new model gets a **fresh budget**.
7. Classifier refusals never consume the exhausted-budget last resort.

**Usage-aware preflight** (`session/agent-session.ts:3014, 4010` `#maybeApplyUsageAwareFallback`): before
spending, when `retry.modelFallback && retry.usageAwareFallback`, consult
`ModelUsageHealth {state:"healthy"|"reserve"|"depleted"|"unknown", accounts[]}` and apply `usageReservePolicy`
(`confirm` = interactive hold / background auto, `auto`, `fail-closed`).

Related surfaces: model-hub UI editor `modes/components/model-hub.ts` (`onFallbackChainChange(role, chain)`);
subagent chain install `task/executor.ts:installSubagentRetryFallbackChain`; security scans explicitly disable
all three tiers (`security/coordinator.ts:228-231`).

---

## 6. Dialect layer — normalising provider API shapes

**Two distinct normalisation layers — do not conflate them.**

### 6.1 Wire-protocol normalisation (`packages/ai/src/providers/` + `stream.ts`)

Each `KnownApi` has a client mapping the canonical `Context`/`AssistantMessageEventStream` onto a vendor HTTP
shape: `anthropic-messages.ts`, `openai-responses.ts`, `openai-completions.ts`, `openai-codex-responses.ts`,
`google-gemini-cli.ts`, `google-vertex.ts`, `amazon-bedrock.ts`, `cursor.ts`, `devin.ts`, `ollama.ts`, plus
`transform-messages.ts` and `openai-shared.ts`. Per-model quirks live in the catalog `compat` block
(`packages/catalog/src/types.ts`): `supportsStrictMode`, `replayUnsignedThinking`, `replayReasoningContent`,
`promptCacheBreakpointTtl`, `disableReasoningOnToolChoice`, `openRouterRouting`, `vercelGatewayRouting`,
`extraBody` — mostly auto-detected from provider id + baseUrl via `packages/catalog/src/hosts.ts`
(`modelMatchesHost`, `hostMatchesUrl`, `isVertexExpressOpenAIUrl`, `isAzureDeploymentsUrl`,
`isDashscopeCompatibleModeUrl`).

### 6.2 `packages/ai/src/dialect/` — **in-band tool-call syntax**

For models without native tool calling: the dialect layer renders the prompt-side syntax AND parses the
stream back into structured events.

```ts
export interface DialectDefinition {
  readonly dialect: CatalogDialect;
  readonly prompt: string;                                  // system-prompt fragment (.md sibling)
  createScanner(options?: InbandScannerOptions): InbandScanner;
  renderToolCall(call, options?): string;                   // inner element only
  renderAssistantToolCalls(calls, options?): string;        // full block incl. envelope
  renderToolResults(results, options?): string;
  renderThinking(text: string): string;
  renderTranscript(messages, options?): string;
}
export interface InbandScanner { feed(text): InbandScanEvent[]; flush(): InbandScanEvent[] }
export type InbandScanEvent =
  | {type:"text";text} | {type:"thinkingStart"} | {type:"thinkingDelta";delta}
  | {type:"thinkingEnd";thinking} | {type:"toolStart";id;name}
  | {type:"toolArgDelta";id;name;key;delta}
  | {type:"toolEnd";id;name;arguments;rawBlock?};
export interface InbandScannerOptions {
  stringArgs?(toolName): ReadonlySet<string>;   // verbatim string args
  tools?: readonly InbandTool[];                // schema-driven dialects
  xmlTagset?: "anthropic" | "dsml";
  parseThinking?: boolean;
}
```

**Registered dialects** (`dialect/factory.ts`, each with a `.ts` plus a prompt `.md`):
`glm`, `hermes`, `kimi`, `xml`, `anthropic`, `deepseek`, `minimax`, `harmony`, `qwen3`, `gemini`, `gemma`.
```ts
getDialectDefinition(dialect): DialectDefinition
createInbandScanner(dialect, options): InbandScanner
```

Support modules: `rendering.ts` (`pyCall`/`pyValue` Python-literal rendering, with verbatim triple-quoted
blocks for multiline payloads when fence-safe; `kimiCallId`, `harmonyRecipient`, `renderToolResponseResults`),
`inventory.ts` (`renderToolInventory` — OpenAI-Harmony `namespace functions { type foo = (_: {...}); }`
catalog), `examples.ts`, `thinking.ts`, `fenced-thinking.ts`, `coercion.ts`, `demotion.ts`, `history.ts`,
`catalog.ts`, `owned-stream.ts`. Stream healing lives alongside in `utils/stream-markup-healing.ts`,
`harmony-leak.ts`, `leaked-thinking-stream.ts`, `thinking-loop.ts`, `tool-call-loop-guard.ts`.

### 6.3 auth-gateway — outward normalisation

`packages/ai/src/auth-gateway/` inverts the layer: it *accepts* provider-format HTTP (OpenAI
chat-completions / Anthropic messages / OpenAI Responses), dispatches through `streamSimple()`, and
re-encodes the canonical stream back to the caller's format, injecting `Authorization` server-side so clients
never see tokens. Default bind `127.0.0.1:4000`.
```ts
export interface AuthGatewayFormatModule {
  parseRequest(body, headers?): AuthGatewayParsedRequest;
  encodeResponse(message, requestedModelId): Record<string,unknown>;
  encodeStream(events, requestedModelId, options?, control?): ReadableStream<Uint8Array>;
  formatError(status, type, message): Response;   // OpenAI vs Anthropic envelopes
}
```
`AuthGatewayParsedRequestOptions` is the canonical union of every sampling/tool/reasoning/routing knob
(`maxOutputTokens, temperature, topP, topK, minP, stopSequences, presencePenalty, frequencyPenalty,
repetitionPenalty, seed, logitBias, responseFormat, toolChoice, parallelToolCalls, include, reasoning,
disableReasoning, explicitThinkingBudgetTokens, thinkingBudgets, hideThinkingSummary, taskBudget,
serviceTier, cacheRetention, promptCacheKey, previousResponseId, user, metadata, headers, extra`) — the best
target schema for a Python request model.

---

## Porting notes to Python

### Library mapping

| omp (Bun/TS) | Python |
|---|---|
| `fetch` / `FetchImpl` injection | **`httpx.AsyncClient`**; inject the client instead of a `fetch` fn (same DI benefit, mockable via `httpx.MockTransport`) |
| `AbortSignal` / `AbortSignal.timeout/any` | `asyncio.CancelledError` + `asyncio.timeout()`; a small `AbortSignal`-shaped wrapper over `asyncio.Event` keeps the port 1:1 |
| `Bun.serve` loopback callback | **`uvicorn` + Starlette**, or stdlib `http.server.ThreadingHTTPServer` for a dependency-free ~40-line callback server. Bind `127.0.0.1`, port 0 for fallback, read `server.server_address[1]` |
| `crypto.getRandomValues` / `crypto.subtle.digest("SHA-256")` | `secrets.token_bytes(96)` + `hashlib.sha256`; `base64.urlsafe_b64encode(...).rstrip(b"=")` |
| `bun:sqlite` | stdlib `sqlite3` (`PRAGMA journal_mode=WAL`, `synchronous=NORMAL`, `busy_timeout`) or `aiosqlite`. The triggers/temp-triggers port verbatim |
| `Bun.hash.xxHash32(sessionId)` | `xxhash.xxh32_intdigest` (pypi `xxhash`), or `zlib.crc32` if a different sticky distribution is acceptable |
| `Bun.sleep` / `scheduler.wait(ms,{signal})` | `await asyncio.sleep()` inside `asyncio.timeout`/`wait_for` |
| JWT claim decode (`decodeJwt`) | manual `base64.urlsafe_b64decode` + `json.loads` — **do not** add `PyJWT`; omp never verifies signatures here and neither should you (the IdP already did) |
| Zod / `@oh-my-pi/omptype` wire schemas | **`pydantic` v2** models for `OAuthCredentials`, `UsageReport`, broker wire types |
| SSE (`GET /v1/snapshot/stream`) | `httpx` streaming + `httpx-sse` client side; `sse-starlette` server side |
| `YAML.parse` (Bun) | `ruamel.yaml` or `pyyaml` |
| `Buffer.from(x,"base64url")` | `base64.urlsafe_b64decode(x + "=" * (-len(x) % 4))` |
| OS keychain (omp uses **none**) | optional upgrade: `keyring` for a DB encryption key while credentials stay in SQLite |

### Flows that need a local callback HTTP server

**Required** (authorization code + PKCE, loopback redirect):

| Provider | Port | Path | Port fallback allowed? |
|---|---|---|---|
| `anthropic` | 54545 | `/callback` | yes |
| `openai-codex` | **1455** | `/auth/callback` | **no** — pinned `redirect_uri` |
| `google-gemini-cli` | 8085 | `/oauth2callback` | yes (host `127.0.0.1`) |
| `google-antigravity` | 51121 | `/oauth-callback` | yes |
| `devin` | 59653 | `/callback` | yes |
| `gitlab-duo` | 8080 | `/callback` | no (GitLab validates) |
| `zai-coding-plan` | 54548 | `/callback` | yes |

**No server needed** (device code / poll / paste): `openai-codex-device`, `kimi-code`, `xai-oauth`,
`github-copilot`, `cursor`, `gitlab-duo-workflow` (`vscode://`), `perplexity`, and every
`createApiKeyLogin` provider.

Port a single abstract `OAuthCallbackFlow` base (subclass hooks `generate_auth_url` / `exchange_token`) and a
single `poll_device_code_flow` coroutine, exactly as omp does — each provider then reduces to ~40 lines of constants.

### Recommended porting order

1. `OAuthCredentials` pydantic model + SQLite `AuthCredentialStore` (schema copies verbatim).
2. `OAuthCallbackFlow` (loopback, port-0 fallback, `/launch` 302 route) + `poll_device_code_flow`.
3. Anthropic + OpenAI Codex (browser **and** device) — these exercise PKCE, pinned redirect_uri, JWT identity,
   and `storeCredentialsAs` aliasing.
4. `ProviderDefinition` registry with lazy imports (Python: module-path string + `importlib.import_module`, or a
   `@register("id")` decorator).
5. `get_api_key` 7-step cascade + round-robin/sticky selection.
6. The a/b/c `auth-retry` policy.
7. Model fallback chains — only if multi-model routing is needed; fully separable.

### Traps worth carrying over verbatim

- **Start the callback server before building the auth URL** so the real bound port lands in `redirect_uri`.
- **Pin `redirect_uri` (disable port fallback) for providers that validate it**, and fail *before* opening the
  browser — otherwise the user gets an opaque 500 and a 5-minute hang.
- **Only offer the paste-code prompt for `pasteCodeFlow` providers**, else the prompt races the HTTP callback
  and leaves the terminal blocked.
- **Never rewrite `orgId`/`orgName` on refresh** — org scope is fixed at login; refresh results are merged
  *over* the stored credential.
- **Skip force-refresh on 403 / usage-limit** — refreshing a valid-but-denied token cannot help.
- **Track attempted bearers in a set** and cap total attempts (omp: 64) or sibling rotation loops forever.
- Anthropic grants die **30 days after authorization** regardless of refresh rotation — surface the deadline.
- Expiry skew: 5 min at mint (`expires_in*1000 - 300000`), 60 s pre-emptive refresh trigger.
- Store the SQLite credential file `0600` and mkdir the parent; Kimi's device-id file is the template.
- A usage endpoint returning 200 does **not** prove the chat endpoint accepts the same bearer — keep the
  separate `CompletionProbe`.

