/* Site approvals on a headless mesh node, decided by the operator's signature.
 *
 * WHY THIS EXISTS. A node-managed Chrome (`lop browser setup --node`) runs with
 * no screen and nobody at the popup, so the ordinary "click Allow" can never
 * happen there. The operator decides on THEIR device instead: the node's session
 * raises a queued ask, the operator answers it in the Local Operator app, and
 * that answer arrives here as `submit_signed_decision` carrying the operator's
 * ES256 signature over the runtime's existing `decide` message.
 *
 * WHY THE EXTENSION VERIFIES (and not only the runtime). The agent holds the
 * bridge token and calls this extension directly. If the extension accepted an
 * unsigned decision, the agent could approve its own request with one bridge
 * call. So nothing short of a signature from the operator anchor — the public
 * point `lop browser setup --node` copies from the root-owned
 * `/etc/local-operator/operators/<uid>.json` into managed policy — decides an
 * entry here. On a node whose user has passwordless sudo that is an advisory,
 * audited boundary (rewriting root-owned policy is visible), not a hard one;
 * the operator accepted that ("leave it, disclose it").
 *
 * WHAT A SIGNATURE BINDS. The signed message is `verify.py`'s
 * `signed_message("decide", session_id, f"{ask_id}#{question_id}#{word}",
 * challenge)`. `question_id` is `site:v1:` + sha256(lp(node) ‖ lp(entryId) ‖
 * lp(origin) ‖ lp(scope)), recomputed HERE from the extension's own live queue
 * entry — so a signature decides exactly one entry, for exactly the origin the
 * entry names, on exactly this node, once (the entry is consumed by the
 * decision and expires on its own 10-minute TTL). The runtime separately
 * single-uses the challenge.
 *
 * Refusals are `ok: true` results with `applied: false` and a `reason`, never
 * new bridge error codes: an already-released daemon validates error codes
 * against its enum and drops a frame carrying one it does not know
 * (protocol.py, above EXTENSION_UNRESPONSIVE).
 *
 * Twins: `local_operator/browser_bridge/site_approval.py` (question id, policy
 * keys) and `local_operator/operator/verify.py` (message layout). The golden
 * vector in tests/signed-decision.test.mjs and
 * tests/unit/browser_bridge/test_site_approval.py pins all three to one set of
 * bytes. */

/** Managed-policy keys. Mirror `site_approval.py` and managed_schema.json. */
export const POLICY_HEADLESS_NODE = "headlessNode";
export const POLICY_ANCHOR_POINT = "operatorAnchorPoint";
export const POLICY_KEY_ID = "operatorKeyId";
export const POLICY_NODE_DEVICE_ID = "nodeDeviceId";

export const SITE_QUESTION_PREFIX = "site:v1:";
/** The only scope a node approval grants: the exact origin (D3). */
export const SITE_SCOPE = "origin";
const DOMAIN = "lop-operator-v1\0";

export type SignedDecisionRefusal =
  | "no_approver"
  | "unsigned_decision"
  | "bad_word"
  | "wrong_key"
  | "no_live_request"
  | "bad_signature";

export interface NodeApproval {
  /** Policy says this is a headless node: nobody can click the popup. */
  headless: boolean;
  /** The anchor, when policy carries a self-consistent one. */
  anchor?: { point: Uint8Array; keyId: string };
  nodeDeviceId: string;
}

const encoder = new TextEncoder();

export function lengthPrefix(value: string): Uint8Array {
  const raw = encoder.encode(value);
  const out = new Uint8Array(4 + raw.length);
  new DataView(out.buffer).setUint32(0, raw.length, false);
  out.set(raw, 4);
  return out;
}

function concat(parts: Uint8Array[]): Uint8Array {
  const out = new Uint8Array(parts.reduce((sum, part) => sum + part.length, 0));
  let offset = 0;
  for (const part of parts) {
    out.set(part, offset);
    offset += part.length;
  }
  return out;
}

export function bytesToHex(bytes: Uint8Array): string {
  let out = "";
  for (const byte of bytes) out += byte.toString(16).padStart(2, "0");
  return out;
}

/** Strict hex: an odd length or a non-hex digit is `undefined`, never a
 * partially-decoded buffer that would verify against the wrong bytes. */
export function hexToBytes(hex: string): Uint8Array | undefined {
  if (typeof hex !== "string" || hex.length % 2 !== 0 || !/^[0-9a-fA-F]*$/.test(hex)) return undefined;
  const out = new Uint8Array(hex.length / 2);
  for (let index = 0; index < out.length; index += 1) {
    out[index] = Number.parseInt(hex.slice(index * 2, index * 2 + 2), 16);
  }
  return out;
}

async function sha256(bytes: Uint8Array): Promise<Uint8Array> {
  return new Uint8Array(await crypto.subtle.digest("SHA-256", bytes as BufferSource));
}

/** `verify.py`'s `signed_message`. */
export function signedMessageBytes(input: {
  action: string;
  sessionId: string;
  requestId: string;
  challenge: string;
}): Uint8Array {
  return concat([
    encoder.encode(DOMAIN),
    lengthPrefix(input.action),
    lengthPrefix(input.sessionId),
    lengthPrefix(input.requestId),
    lengthPrefix(input.challenge),
  ]);
}

/** `site_approval.site_question_id`. */
export async function siteQuestionId(input: {
  nodeDeviceId: string;
  entryId: string;
  origin: string;
  scope?: string;
}): Promise<string> {
  const digest = await sha256(
    concat([
      lengthPrefix(input.nodeDeviceId),
      lengthPrefix(input.entryId),
      lengthPrefix(input.origin),
      lengthPrefix(input.scope ?? SITE_SCOPE),
    ]),
  );
  return SITE_QUESTION_PREFIX + bytesToHex(digest);
}

/** `verify.py`'s `key_id_for`: the first 32 hex chars of sha256(point). */
export async function keyIdFor(point: Uint8Array): Promise<string> {
  return bytesToHex(await sha256(point)).slice(0, 32);
}

/** DER `SEQUENCE { INTEGER r, INTEGER s }` → raw 64-byte r‖s for WebCrypto.
 *
 * Strict, because the input is attacker-supplied: every length is checked
 * against the buffer, the sequence must consume the whole signature, and each
 * integer must fit 32 bytes once its sign-padding zero is dropped. Anything
 * else is `undefined` (a refusal), never a best-effort guess. */
export function derSignatureToRaw(der: Uint8Array): Uint8Array | undefined {
  if (der.length < 8 || der.length > 80 || der[0] !== 0x30) return undefined;
  if (der[1] !== der.length - 2) return undefined;
  const out = new Uint8Array(64);
  let offset = 2;
  for (const half of [0, 1]) {
    if (offset + 2 > der.length || der[offset] !== 0x02) return undefined;
    const length = der[offset + 1] ?? 0;
    const start = offset + 2;
    const end = start + length;
    if (length < 1 || end > der.length) return undefined;
    let value = der.subarray(start, end);
    while (value.length > 1 && value[0] === 0x00) value = value.subarray(1);
    if (value.length > 32) return undefined;
    out.set(value, half * 32 + (32 - value.length));
    offset = end;
  }
  return offset === der.length ? out : undefined;
}

/** Read the node-approval block from `chrome.storage.managed` (or a test
 * double). An anchor whose `operatorKeyId` is not the digest of its point is
 * dropped — a hand-edited pair that disagrees fails CLOSED (no approver), it
 * never verifies against "whichever of the two looked right". */
export async function readNodeApproval(
  managed: Pick<chrome.storage.StorageArea, "get"> | undefined,
): Promise<NodeApproval> {
  if (!managed) return { headless: false, nodeDeviceId: "" };
  let raw: Record<string, unknown> = {};
  try {
    raw = ((await managed.get([
      POLICY_HEADLESS_NODE,
      POLICY_ANCHOR_POINT,
      POLICY_KEY_ID,
      POLICY_NODE_DEVICE_ID,
    ])) ?? {}) as Record<string, unknown>;
  } catch {
    return { headless: false, nodeDeviceId: "" };
  }
  const headless = raw[POLICY_HEADLESS_NODE] === true;
  const nodeDeviceId = typeof raw[POLICY_NODE_DEVICE_ID] === "string" ? (raw[POLICY_NODE_DEVICE_ID] as string) : "";
  const point = typeof raw[POLICY_ANCHOR_POINT] === "string" ? hexToBytes(raw[POLICY_ANCHOR_POINT] as string) : undefined;
  const keyId = typeof raw[POLICY_KEY_ID] === "string" ? (raw[POLICY_KEY_ID] as string) : "";
  if (!point || point.length !== 65 || point[0] !== 0x04 || !keyId) return { headless, nodeDeviceId };
  if ((await keyIdFor(point)) !== keyId) return { headless, nodeDeviceId };
  return { headless, anchor: { point, keyId }, nodeDeviceId };
}

/** The live queue entry a signed decision may name. */
export interface DecidableEntry {
  entryId: string;
  origin: string;
  expiresAt: number;
}

export interface SignedDecisionDeps {
  approval: NodeApproval;
  /** The extension's LIVE (unexpired, undecided) queue entries. */
  liveEntries: () => Promise<DecidableEntry[]>;
  /** Apply the decision; `false` when the entry was decided/expired meanwhile. */
  decide: (entryId: string, decision: "site" | "deny") => Promise<boolean>;
  now?: () => number;
}

export interface SignedDecisionResult {
  applied: boolean;
  reason?: SignedDecisionRefusal;
  origin?: string;
  state?: "allowed" | "denied";
}

function text(value: unknown): string {
  return typeof value === "string" ? value : "";
}

/** Verify one operator decision and apply it to exactly the entry it binds.
 * The order puts the cheap, non-secret refusals first and the signature last,
 * but no refusal path reveals anything a caller could use to forge one. */
export async function submitSignedDecisionWith(
  params: Record<string, unknown>,
  deps: SignedDecisionDeps,
): Promise<SignedDecisionResult> {
  const anchor = deps.approval.anchor;
  if (!anchor) return { applied: false, reason: "no_approver" };
  const sig = text(params.sig);
  const challenge = text(params.challenge);
  const keyId = text(params.key_id);
  if (!sig || !challenge || !keyId) return { applied: false, reason: "unsigned_decision" };
  const word = text(params.word).trim().toLowerCase();
  if (word !== "approve" && word !== "deny") return { applied: false, reason: "bad_word" };
  if (keyId !== anchor.keyId) return { applied: false, reason: "wrong_key" };
  const questionId = text(params.question_id);
  if (!questionId.startsWith(SITE_QUESTION_PREFIX)) return { applied: false, reason: "no_live_request" };
  const now = (deps.now ?? Date.now)();
  let entry: DecidableEntry | undefined;
  for (const candidate of await deps.liveEntries()) {
    if (now >= candidate.expiresAt) continue;
    const derived = await siteQuestionId({
      nodeDeviceId: deps.approval.nodeDeviceId,
      entryId: candidate.entryId,
      origin: candidate.origin,
    });
    if (derived === questionId) {
      entry = candidate;
      break;
    }
  }
  if (!entry) return { applied: false, reason: "no_live_request" };
  const der = hexToBytes(sig);
  const raw = der ? derSignatureToRaw(der) : undefined;
  if (!raw) return { applied: false, reason: "bad_signature" };
  const message = signedMessageBytes({
    action: "decide",
    sessionId: text(params.session_id),
    requestId: `${text(params.ask_id)}#${questionId}#${word}`,
    challenge,
  });
  let valid = false;
  try {
    const key = await crypto.subtle.importKey(
      "raw",
      anchor.point as BufferSource,
      { name: "ECDSA", namedCurve: "P-256" },
      false,
      ["verify"],
    );
    valid = await crypto.subtle.verify(
      { name: "ECDSA", hash: "SHA-256" },
      key,
      raw as BufferSource,
      message as BufferSource,
    );
  } catch {
    valid = false;
  }
  if (!valid) return { applied: false, reason: "bad_signature" };
  const decision = word === "approve" ? "site" : "deny";
  if (!(await deps.decide(entry.entryId, decision))) return { applied: false, reason: "no_live_request" };
  return { applied: true, origin: entry.origin, state: decision === "site" ? "allowed" : "denied" };
}
