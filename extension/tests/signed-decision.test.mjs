/* Headless-node signed site approval (src/signed-decision.ts).
 *
 * Pins the contract the session runtime and `site_approval.py` rely on: the
 * question id and the signed `decide` message are byte-identical to Python's
 * (the golden vector is SHARED with tests/unit/browser_bridge/
 * test_site_approval.py), a DER signature from the policy anchor decides
 * exactly the entry its question id binds — once — and every other shape is
 * refused with a typed reason: unsigned, wrong key, a signature over different
 * bytes, a consumed or expired entry, a policy with no (or an inconsistent)
 * anchor. */
import { test } from "node:test";
import assert from "node:assert/strict";
import { build } from "esbuild";
import { mkdtemp, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const VECTOR = JSON.parse(await readFile(join(HERE, "fixtures", "site-approval-vector.json"), "utf8"));

async function load() {
  const dir = await mkdtemp(join(tmpdir(), "lop-signed-decision-"));
  const outfile = join(dir, "signed.mjs");
  await build({
    entryPoints: [join(HERE, "..", "src", "signed-decision.ts")],
    bundle: true,
    platform: "node",
    format: "esm",
    outfile,
  });
  return import(pathToFileURL(outfile) + `?${Date.now()}`);
}

const mod = await load();

function policy(overrides = {}) {
  const values = {
    headlessNode: true,
    operatorAnchorPoint: VECTOR.anchor_point,
    operatorKeyId: VECTOR.anchor_key_id,
    nodeDeviceId: VECTOR.node_device_id,
    ...overrides,
  };
  return { get: async () => values };
}

function params(overrides = {}) {
  return {
    session_id: VECTOR.session_id,
    ask_id: VECTOR.ask_id,
    question_id: VECTOR.question_id,
    word: VECTOR.word,
    challenge: VECTOR.challenge,
    sig: VECTOR.sig_der,
    key_id: VECTOR.anchor_key_id,
    ...overrides,
  };
}

/** A live queue of one entry (the vector's), consumed by the first decision. */
function harness({ expiresAt = Date.now() + 60_000 } = {}) {
  const entries = [{ entryId: VECTOR.entry_id, origin: VECTOR.origin, expiresAt }];
  const decisions = [];
  return {
    decisions,
    liveEntries: async () => entries.slice(),
    decide: async (entryId, decision) => {
      const index = entries.findIndex((entry) => entry.entryId === entryId);
      if (index < 0) return false;
      entries.splice(index, 1);
      decisions.push([entryId, decision]);
      return true;
    },
  };
}

async function submit(overrides, { policyOverrides, h = harness() } = {}) {
  const approval = await mod.readNodeApproval(policy(policyOverrides));
  const result = await mod.submitSignedDecisionWith(params(overrides), {
    approval,
    liveEntries: h.liveEntries,
    decide: h.decide,
  });
  return { result, h };
}

test("golden vector: question id, signed message and key id match Python byte for byte", async () => {
  const questionId = await mod.siteQuestionId({
    nodeDeviceId: VECTOR.node_device_id,
    entryId: VECTOR.entry_id,
    origin: VECTOR.origin,
    scope: VECTOR.scope,
  });
  assert.equal(questionId, VECTOR.question_id);
  const message = mod.signedMessageBytes({
    action: "decide",
    sessionId: VECTOR.session_id,
    requestId: `${VECTOR.ask_id}#${VECTOR.question_id}#${VECTOR.word}`,
    challenge: VECTOR.challenge,
  });
  assert.equal(mod.bytesToHex(message), VECTOR.message_hex);
  assert.equal(await mod.keyIdFor(mod.hexToBytes(VECTOR.anchor_point)), VECTOR.anchor_key_id);
});

test("DER → raw: 64 bytes, verifies under WebCrypto; malformed DER is refused", async () => {
  const raw = mod.derSignatureToRaw(mod.hexToBytes(VECTOR.sig_der));
  assert.equal(raw.length, 64);
  const key = await crypto.subtle.importKey(
    "raw", mod.hexToBytes(VECTOR.anchor_point), { name: "ECDSA", namedCurve: "P-256" }, false, ["verify"],
  );
  assert.ok(await crypto.subtle.verify(
    { name: "ECDSA", hash: "SHA-256" }, key, raw, mod.hexToBytes(VECTOR.message_hex),
  ));
  const der = mod.hexToBytes(VECTOR.sig_der);
  assert.equal(mod.derSignatureToRaw(der.subarray(0, der.length - 1)), undefined, "truncated");
  assert.equal(mod.derSignatureToRaw(new Uint8Array([0x31, ...der.subarray(1)])), undefined, "not a sequence");
  assert.equal(mod.derSignatureToRaw(new Uint8Array([...der, 0x00])), undefined, "trailing byte");
});

test("a good signature from the anchor approves exactly its entry, as an exact-origin grant", async () => {
  const { result, h } = await submit({});
  assert.deepEqual(result, { applied: true, origin: VECTOR.origin, state: "allowed" });
  assert.deepEqual(h.decisions, [[VECTOR.entry_id, "site"]]);
});

test("unsigned decisions are refused (the node's own console holds no operator key)", async () => {
  for (const missing of [{ sig: "" }, { challenge: "" }, { key_id: "" }]) {
    const { result, h } = await submit(missing);
    assert.deepEqual(result, { applied: false, reason: "unsigned_decision" });
    assert.deepEqual(h.decisions, []);
  }
});

test("a signature from a key that is not the anchor is refused", async () => {
  // Named as itself: refused before any crypto.
  const named = await submit({ sig: VECTOR.other_sig_der, key_id: VECTOR.other_key_id });
  assert.equal(named.result.reason, "wrong_key");
  // Claiming the anchor's key id: the signature itself does not verify.
  const claimed = await submit({ sig: VECTOR.other_sig_der });
  assert.equal(claimed.result.reason, "bad_signature");
  assert.deepEqual(claimed.h.decisions, []);
});

test("a signature over different bytes (word, session, ask, challenge) is refused", async () => {
  for (const changed of [
    { word: "deny" },
    { session_id: "another-session" },
    { ask_id: "ask_other" },
    { challenge: "d".repeat(64) },
  ]) {
    const { result, h } = await submit(changed);
    assert.equal(result.reason, "bad_signature", JSON.stringify(changed));
    assert.deepEqual(h.decisions, []);
  }
});

test("a question id for another entry, origin or node decides nothing", async () => {
  const other = await mod.siteQuestionId({
    nodeDeviceId: VECTOR.node_device_id, entryId: VECTOR.entry_id, origin: "https://evil.example",
  });
  assert.equal((await submit({ question_id: other })).result.reason, "no_live_request");
  // Same entry, but this extension sits on a different node: the digest differs.
  const elsewhere = await submit({}, { policyOverrides: { nodeDeviceId: "d_ffffffffffffffffffffffffffffffff" } });
  assert.equal(elsewhere.result.reason, "no_live_request");
});

test("replay: the consumed entry cannot be decided twice", async () => {
  const h = harness();
  const first = await submit({}, { h });
  assert.equal(first.result.applied, true);
  const replay = await submit({}, { h });
  assert.deepEqual(replay.result, { applied: false, reason: "no_live_request" });
  assert.equal(h.decisions.length, 1);
});

test("an expired entry is refused even if still listed", async () => {
  const { result, h } = await submit({}, { h: harness({ expiresAt: Date.now() - 1 }) });
  assert.deepEqual(result, { applied: false, reason: "no_live_request" });
  assert.deepEqual(h.decisions, []);
});

test("no anchor, or an anchor whose key id is not its point's digest, means no approver", async () => {
  assert.equal((await submit({}, { policyOverrides: { operatorAnchorPoint: undefined } })).result.reason, "no_approver");
  assert.equal(
    (await submit({}, { policyOverrides: { operatorKeyId: VECTOR.other_key_id } })).result.reason,
    "no_approver",
  );
  // Swapping in another point while keeping the anchor's id also fails closed.
  assert.equal(
    (await submit({}, { policyOverrides: { operatorAnchorPoint: VECTOR.other_point } })).result.reason,
    "no_approver",
  );
  const none = await mod.readNodeApproval(undefined);
  assert.deepEqual(none, { headless: false, nodeDeviceId: "" });
});

test("bad words are refused before any lookup", async () => {
  assert.equal((await submit({ word: "allow" })).result.reason, "bad_word");
});

test("managed_schema.json declares every key the reader asks for", async () => {
  const schema = JSON.parse(await readFile(join(HERE, "..", "managed_schema.json"), "utf8"));
  for (const key of [mod.POLICY_HEADLESS_NODE, mod.POLICY_ANCHOR_POINT, mod.POLICY_KEY_ID, mod.POLICY_NODE_DEVICE_ID]) {
    assert.ok(schema.properties[key], `${key} must be declared or Chrome never delivers it`);
  }
});
