import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { promisify } from "node:util";

const run = promisify(execFile);
// The expected version is DERIVED from package.json rather than hardcoded so a
// release bump is a two-file change (manifest + package) that cannot silently
// break these tests: every fixture below follows the real version, and the
// mismatch guard keeps using a literal impossible version (9.9.9).
const VERSION = JSON.parse(
  await readFile(new URL("../package.json", import.meta.url), "utf8"),
).version;
const extensionId = "omibaecbjdhgbbcedbnnnmjpmopfheof";
const itemPath = `/v2/publishers/test-publisher/items/${extensionId}`;

// A handler normally returns the JSON payload of a 200. Wrapping it in
// `rejectWith` makes the stub answer with a non-2xx and that body instead,
// which is how the store reports a refused upload or publish.
const rejectWith = (status, body) => ({ __rejectStatus: status, body });

async function runRelease(args, handlers, options = {}) {
  const { token = "test-token", expectFailure = false } = options;
  let requestIndex = 0;
  const server = createServer(async (request, response) => {
    const chunks = [];
    for await (const chunk of request) chunks.push(chunk);
    try {
      assert.equal(request.headers.authorization, `Bearer ${token}`);
      const handler = handlers[requestIndex++];
      assert.ok(handler, `unexpected request ${request.method} ${request.url}`);
      const payload = handler(request, Buffer.concat(chunks));
      const rejected = payload && payload.__rejectStatus;
      response.writeHead(rejected ?? 200, { "Content-Type": "application/json" });
      response.end(JSON.stringify(rejected ? payload.body : payload));
    } catch (error) {
      response.writeHead(500, { "Content-Type": "application/json" });
      response.end(JSON.stringify({ error: String(error) }));
    }
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address();
  const options_ = {
    cwd: import.meta.dirname + "/..",
    env: {
      ...process.env,
      CWS_API_ROOT: `http://127.0.0.1:${port}`,
      CWS_ACCESS_TOKEN: token,
      CWS_PUBLISHER_ID: "test-publisher",
      CWS_EXTENSION_ID: extensionId,
      CWS_POLL_INTERVAL_SECONDS: "0",
    },
  };
  try {
    let result;
    if (expectFailure) {
      // The script must exit non-zero here; a run that SUCCEEDS is itself the
      // failure, so it is reported rather than silently accepted.
      result = await run("bash", ["scripts/chrome-web-store.sh", ...args], options_)
        .then((ok) => { throw new Error(`expected a non-zero exit, got:\n${ok.stdout}`); },
          (error) => error);
    } else {
      result = await run("bash", ["scripts/chrome-web-store.sh", ...args], options_);
    }
    assert.equal(requestIndex, handlers.length);
    return result;
  } finally {
    await new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
  }
}

test("manifest requests tabGroups exactly once", async () => {
  const manifest = JSON.parse(await readFile(new URL("../manifest.json", import.meta.url), "utf8"));
  // The invariant is that the two manifests AGREE, not that they sit at one
  // hardcoded number: the store package is built from manifest.json while every
  // release script reads package.json, so a bump applied to only one of them
  // ships a zip whose version does not match what was promoted. A literal here
  // just has to be edited on every release, and says nothing when it is.
  assert.equal(manifest.version, VERSION, "manifest.json and package.json must carry the same version");
  assert.match(manifest.version, /^\d+\.\d+\.\d+$/, "the manifest version must be a plain semver triple");
  assert.equal(manifest.permissions.filter((permission) => permission === "tabGroups").length, 1);
});

test("store zip is allowlisted, source-map-free, and version-aligned", async () => {
  await run("node", ["build.mjs", "--zip"], { cwd: import.meta.dirname + "/.." });
  const validated = await run("bash", ["scripts/validate-store-zip.sh", "local-operator-extension.zip", VERSION], {
    cwd: import.meta.dirname + "/..",
  });
  assert.match(validated.stdout, new RegExp(`validated Chrome Web Store package v${VERSION.replaceAll(".", "\\.")}`));
  await assert.rejects(
    run("bash", ["scripts/validate-store-zip.sh", "local-operator-extension.zip", "9.9.9"], {
      cwd: import.meta.dirname + "/..",
    }),
    /does not match expected version 9\.9\.9/,
  );
});

test("publisher refuses any item except the permanent extension ID", async () => {
  await assert.rejects(
    run("bash", ["scripts/chrome-web-store.sh", "promote", VERSION], {
      cwd: import.meta.dirname + "/..",
      env: {
        ...process.env,
        CWS_ACCESS_TOKEN: "test-token",
        CWS_PUBLISHER_ID: "test-publisher",
        CWS_EXTENSION_ID: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      },
    }),
    /must be the permanent Local Operator ID/,
  );
});

// Run 34178951112 failed with nothing but `curl: (22) The requested URL
// returned error: 400`. The store's explanation HAD been downloaded -- curl's
// --fail-with-body writes the body and still exits 22, so `set -e` killed the
// script before anything read the file. These tests pin that the body reaches
// the release owner, because the log line alone left no way to tell whether to
// wait, retry, or open the dashboard.
const IN_REVIEW_BODY = {
  error: {
    code: 400,
    message: "Item is currently in review and cannot be updated.",
    status: "FAILED_PRECONDITION",
  },
};

for (const { name, args, handlers, label } of [
  {
    name: "upload",
    label: "upload",
    args: ["stage", "local-operator-extension.zip", VERSION],
    handlers: [() => rejectWith(400, IN_REVIEW_BODY)],
  },
  {
    name: "publish",
    label: "publish",
    args: ["stage", "local-operator-extension.zip", VERSION],
    handlers: [
      () => ({ itemId: extensionId, uploadState: "SUCCEEDED", crxVersion: VERSION }),
      () => rejectWith(400, IN_REVIEW_BODY),
    ],
  },
  {
    name: "fetchStatus",
    label: "fetchStatus",
    args: ["promote", VERSION],
    handlers: [() => rejectWith(400, IN_REVIEW_BODY)],
  },
]) {
  test(`a rejected ${name} call surfaces the store's explanation`, async () => {
    const error = await runRelease(args, handlers, { expectFailure: true });
    const output = error.stdout + error.stderr;
    // The body itself -- the one thing the incident threw away.
    assert.match(output, /Item is currently in review and cannot be updated\./);
    assert.match(output, /FAILED_PRECONDITION/);
    // Which call failed, at what status, for which version.
    assert.match(output, new RegExp(`${label} call returned HTTP 400`));
    assert.match(output, new RegExp(`v${VERSION.replaceAll(".", "\\.")}`));
    // The remedy a release owner acts on.
    assert.match(output, /cancelSubmission/);
    // curl's bare exit 22 must no longer be the whole story.
    assert.notEqual(error.code, 22);
  });
}

test("a transport failure is named as one, not reported as an HTTP status", async () => {
  // Pins the `rc` guard, which a mutation proved load-bearing but unpinned:
  // deleting it left the whole suite green. It is not redundant with the
  // status check, because curl reports `http_code=200` alongside `rc=18` on a
  // transfer truncated against its Content-Length -- a body that parses as
  // valid JSON while the transfer was in fact broken. Only `rc` catches that.
  // A closed port is the cheap, deterministic form of the same guard.
  const server = createServer(() => {});
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address();
  // Release the port before pointing the script at it, so nothing is listening
  // and the socket cannot collide with an unrelated local service.
  await new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));

  const error = await run("bash", ["scripts/chrome-web-store.sh", "promote", VERSION], {
    cwd: import.meta.dirname + "/..",
    env: {
      ...process.env,
      CWS_API_ROOT: `http://127.0.0.1:${port}`,
      CWS_ACCESS_TOKEN: "test-token",
      CWS_PUBLISHER_ID: "test-publisher",
      CWS_EXTENSION_ID: extensionId,
      CWS_POLL_INTERVAL_SECONDS: "0",
    },
  }).then((ok) => { throw new Error(`expected a non-zero exit, got:\n${ok.stdout}`); }, (e) => e);

  const output = error.stdout + error.stderr;
  // Named as unreachable, carrying curl's own exit code -- NOT dressed up as an
  // HTTP status, which is what a missing `rc` guard would produce (`HTTP 000`).
  assert.match(output, /fetchStatus call could not reach the Chrome Web Store \(curl exit \d+\)/);
  assert.doesNotMatch(output, /returned HTTP/);
});

// Mirrors BODY_PRINT_LIMIT in chrome-web-store.sh. The straddle test below has
// to know where the cut falls, which is the one thing about the bound a test
// cannot discover from the outside.
const BODY_PRINT_LIMIT = 4000;

test("an oversized error body is bounded and its truncation is disclosed", async () => {
  // An unbounded echo floods the run log -- a single 400 was measured at
  // 3,000,626 bytes of step output -- which works against the readability this
  // script exists to deliver. The cut must be visible, so a partial body is
  // never mistaken for the store's complete answer.
  const filler = "F".repeat(200_000);
  const error = await runRelease(
    ["stage", "local-operator-extension.zip", VERSION],
    [() => rejectWith(400, { error: { code: 400, status: "INVALID_ARGUMENT", message: filler } })],
    { expectFailure: true },
  );
  const output = error.stdout + error.stderr;
  assert.match(output, /\(truncated to \d+ of \d+ characters\)/);
  // Bounded well below what the stub sent, while still showing the beginning.
  assert.ok(output.length < 20_000, `expected a bounded log, got ${output.length} characters`);
  assert.match(output, /INVALID_ARGUMENT/);
  // Truncating the body must not cost the remedy that follows it.
  assert.match(output, /cancelSubmission/);
});

test("a token straddling the truncation boundary is redacted, not cut in half", async () => {
  // Pins the ORDER of the redact and truncate blocks, which a comment argues
  // for and nothing enforced: reversing them left the suite 30/30 green while
  // leaking up to 28 characters of the token. Truncating first cuts the token
  // in half, and the surviving prefix no longer matches the substitution
  // pattern, so it prints verbatim -- a partial credential is still a leak.
  const token = "ya29.a0AfB_STRADDLE-SECRET-TOKEN-VALUE";
  const body = (pad) => ({
    error: { code: 400, status: "INVALID_ARGUMENT", message: "P".repeat(pad) + token + "Q".repeat(400) },
  });
  // DERIVE the padding that puts the cut inside the token rather than hardcode
  // it: jq's pretty-printing decides the offset, so a hardcoded pad would stop
  // straddling the moment the envelope changed and the test would keep passing
  // while measuring nothing. Half the token either side of the limit gives the
  // reversed order its longest surviving prefix.
  const envelope = JSON.stringify(body(0), null, 2).indexOf(token);
  const pad = BODY_PRINT_LIMIT - envelope - Math.floor(token.length / 2);
  const rendered = JSON.stringify(body(pad), null, 2);
  const start = rendered.indexOf(token);
  assert.ok(
    start < BODY_PRINT_LIMIT && start + token.length > BODY_PRINT_LIMIT,
    `the token must straddle the cut to measure anything (start ${start}, limit ${BODY_PRINT_LIMIT})`,
  );

  const error = await runRelease(
    ["stage", "local-operator-extension.zip", VERSION],
    [() => rejectWith(400, body(pad))],
    { token, expectFailure: true },
  );
  const output = error.stdout + error.stderr;
  // No prefix of the credential may survive -- not the whole token, and not the
  // leading fragment the cut would otherwise leave behind. Report the surviving
  // prefix itself on failure: the length is the finding, and a slice of the
  // output at a fixed offset would show padding rather than the leak. Safe to
  // print because this token is synthetic and local to the test.
  assert.ok(!output.includes(token), "the whole token must never reach the log");
  let survived = "";
  for (let n = token.length; n >= 1; n -= 1) {
    if (output.includes(token.slice(0, n))) { survived = token.slice(0, n); break; }
  }
  assert.ok(
    !output.includes(token.slice(0, 8)),
    `${survived.length} characters of the token survived truncation: ${JSON.stringify(survived)}`,
  );
  // The body really was cut here; otherwise the assertions above are vacuous.
  assert.match(output, /\(truncated to \d+ of \d+ characters\)/);
});

test("a rejected call does not echo the access token, even if the body carries it", async () => {
  // A response body is NOT masked by Actions, and an API that echoed request
  // context could hand the token straight back. Nothing in this output may
  // carry it: not the body, and not any message the script composes.
  const token = "ya29.a0AfB_bearer-token-shaped-secret-value";
  const error = await runRelease(
    ["stage", "local-operator-extension.zip", VERSION],
    [() => rejectWith(400, {
      error: {
        code: 400,
        status: "INVALID_ARGUMENT",
        message: `rejected request authorized by ${token}`,
      },
    })],
    { token, expectFailure: true },
  );
  const output = error.stdout + error.stderr;
  assert.ok(!output.includes(token), "the access token must never reach the log");
  assert.match(output, /<redacted CWS_ACCESS_TOKEN>/);
  // Redaction must not cost the diagnosis: the rest of the body still shows.
  assert.match(output, /INVALID_ARGUMENT/);
});

test("stage uploads the validated zip and requests deferred publication", async () => {
  const result = await runRelease(["stage", "local-operator-extension.zip", VERSION], [
    (request, body) => {
      assert.equal(request.method, "POST");
      assert.equal(request.url, `/upload${itemPath}:upload`);
      assert.equal(request.headers["content-type"], "application/zip");
      assert.ok(body.length > 1_000);
      return { itemId: extensionId, uploadState: "SUCCEEDED", crxVersion: VERSION };
    },
    (request, body) => {
      assert.equal(request.method, "POST");
      assert.equal(request.url, `${itemPath}:publish`);
      assert.deepEqual(JSON.parse(body), {
        publishType: "STAGED_PUBLISH",
        deployInfos: [{ deployPercentage: 100 }],
        blockOnWarnings: true,
      });
      return { itemId: extensionId, state: "PENDING_REVIEW" };
    },
  ]);
  assert.match(result.stdout, /with STAGED_PUBLISH \(PENDING_REVIEW\)/);
});

test("asynchronous upload fails closed instead of trusting global upload status", async () => {
  await assert.rejects(
    runRelease(["stage", "local-operator-extension.zip", VERSION], [
      (request) => {
        assert.equal(request.method, "POST");
        return { itemId: extensionId, uploadState: "IN_PROGRESS" };
      },
    ]),
    new RegExp(`asynchronous upload cannot be bound to version ${VERSION.replaceAll(".", "\\.")}`),
  );
});

test("promotion verifies the approved version at 100 percent before making it public", async () => {
  const staged = {
    itemId: extensionId,
    submittedItemRevisionStatus: {
      state: "STAGED",
      distributionChannels: [{ crxVersion: VERSION, deployPercentage: 100 }],
    },
  };
  const result = await runRelease(["promote", VERSION], [
    (request) => {
      assert.equal(request.method, "GET");
      assert.equal(request.url, `${itemPath}:fetchStatus`);
      return staged;
    },
    (request, body) => {
      assert.equal(request.method, "POST");
      assert.deepEqual(JSON.parse(body), {
        publishType: "STAGED_PUBLISH",
        deployInfos: [{ deployPercentage: 100 }],
        blockOnWarnings: true,
      });
      return { itemId: extensionId, state: "PUBLISHED" };
    },
    (request) => {
      assert.equal(request.method, "GET");
      return {
        itemId: extensionId,
        publishedItemRevisionStatus: {
          state: "PUBLISHED",
          distributionChannels: [{ crxVersion: VERSION, deployPercentage: 100 }],
        },
      };
    },
  ]);
  assert.match(result.stdout, new RegExp(`v${VERSION.replaceAll(".", "\\.")} to PUBLISHED`));
});

test("promotion refuses a staged revision below 100 percent", async () => {
  await assert.rejects(
    runRelease(["promote", VERSION], [
      () => ({
        itemId: extensionId,
        submittedItemRevisionStatus: {
          state: "STAGED",
          distributionChannels: [{ crxVersion: VERSION, deployPercentage: 50 }],
        },
      }),
    ]),
    new RegExp(`must contain version ${VERSION.replaceAll(".", "\\.")} at 100% deployment`),
  );
});

// The 0.1.12 promotion failed on this gate four more times with a message that
// named only what the gate WANTED. `distributionChannels` describing the LIVE
// revision is the shape that has to be distinguishable, from the log alone,
// from a staged rollout still settling -- so both are pinned below.
test("promotion refusal quotes the revision the store reported", async () => {
  const error = await runRelease(["promote", VERSION], [
    () => ({
      itemId: extensionId,
      submittedItemRevisionStatus: {
        state: "STAGED",
        // The live 0.1.10 revision still sitting in the submitted channels:
        // this script's own history, and the reason the summary prints every
        // field rather than only the one the gate reads.
        distributionChannels: [{ crxVersion: "0.1.10", deployPercentage: 100 }],
      },
    }),
  ], { expectFailure: true });
  const output = error.stdout + error.stderr;
  // The leading text is unchanged, so log greps in the docs and in past runs
  // still match it.
  assert.ok(output.includes(`staged revision must contain version ${VERSION} at 100% deployment`), output);
  assert.ok(
    output.includes("store said: submitted state=STAGED distributionChannels=[crxVersion=0.1.10 deployPercentage=100]"),
    output,
  );
  // A revision the store did not send reads as absent -- a different answer
  // from an empty channel list, which prints as `[]`.
  assert.ok(output.includes("published <absent>"), output);
});

test("promotion refusal reports the staging percentage the store sent", async () => {
  const error = await runRelease(["promote", VERSION], [
    () => ({
      itemId: extensionId,
      submittedItemRevisionStatus: {
        state: "STAGED",
        distributionChannels: [{ crxVersion: VERSION, deployPercentage: 50 }],
      },
    }),
  ], { expectFailure: true });
  const output = error.stdout + error.stderr;
  assert.ok(
    output.includes(`store said: submitted state=STAGED distributionChannels=[crxVersion=${VERSION} deployPercentage=50]`),
    output,
  );
});

test("a submission that is not STAGED reports the state the store sent", async () => {
  const error = await runRelease(["promote", VERSION], [
    () => ({
      itemId: extensionId,
      submittedItemRevisionStatus: {
        state: "PENDING_REVIEW",
        distributionChannels: [{ crxVersion: VERSION, deployPercentage: 100 }],
      },
    }),
  ], { expectFailure: true });
  const output = error.stdout + error.stderr;
  // The first gate's leading text, unchanged, now carrying the store's own
  // state -- the difference between "still in review" and "rejected".
  assert.ok(output.includes("only an approved STAGED revision can be promoted (store said: submitted state=PENDING_REVIEW"), output);
  // runRelease asserts the request count, so the single handler above proves
  // this refusal reached no publish call on its way out.
});

test("the polling deadline reports the last response instead of guessing", async () => {
  // The deadline is the third place the script used to fail blind: the publish
  // call is accepted and the item still never reaches PUBLISHED at 100%.
  const staged = {
    itemId: extensionId,
    submittedItemRevisionStatus: {
      state: "STAGED",
      distributionChannels: [{ crxVersion: VERSION, deployPercentage: 100 }],
    },
  };
  const notYet = {
    itemId: extensionId,
    publishedItemRevisionStatus: {
      state: "STAGED",
      distributionChannels: [{ crxVersion: "0.1.10", deployPercentage: 100 }],
    },
  };
  // One fetchStatus, one publish, then the loop's full twelve polls: this count
  // is the loop's own, so changing the loop has to change it here too.
  const error = await runRelease(["promote", VERSION], [
    () => staged,
    () => ({ itemId: extensionId, state: "PUBLISHED" }),
    ...Array.from({ length: 12 }, () => () => notYet),
  ], { expectFailure: true });
  const output = error.stdout + error.stderr;
  assert.ok(output.includes("was not PUBLISHED at 100% before the polling deadline"), output);
  // The last poll's own fields, so the reader can see WHICH revision the store
  // was still holding instead of being told only what the deadline expected.
  assert.ok(output.includes("last response said:"), output);
  assert.ok(
    output.includes("published state=STAGED distributionChannels=[crxVersion=0.1.10 deployPercentage=100]"),
    output,
  );
});

// `status` exists because reading the queue otherwise means dispatching `stage`,
// which -- while an item is in review -- is refused with HTTP 400
// FAILED_PRECONDITION / NOT_UPDATEABLE. It must be a pure read, and the request
// count runRelease asserts is what pins that no upload or publish follows it.
test("status reads the queue without uploading or publishing", async () => {
  const result = await runRelease(["status"], [
    (request) => {
      assert.equal(request.method, "GET");
      assert.equal(request.url, `${itemPath}:fetchStatus`);
      return {
        itemId: extensionId,
        lastAsyncUploadState: "SUCCEEDED",
        submittedItemRevisionStatus: {
          state: "STAGED",
          distributionChannels: [{ crxVersion: VERSION, deployPercentage: 100 }],
        },
        publishedItemRevisionStatus: {
          state: "PUBLISHED",
          distributionChannels: [{ crxVersion: "0.1.10", deployPercentage: 100 }],
        },
      };
    },
  ]);
  assert.ok(result.stdout.includes(`Chrome Web Store status for extension ${extensionId}:`), result.stdout);
  assert.ok(
    result.stdout.includes(`submitted state=STAGED distributionChannels=[crxVersion=${VERSION} deployPercentage=100]`),
    result.stdout,
  );
  // The same rendering the gates attach, so a `status` read and a refused
  // promotion can be compared line by line.
  assert.ok(
    result.stdout.includes("published state=PUBLISHED distributionChannels=[crxVersion=0.1.10 deployPercentage=100]"),
    result.stdout,
  );
  assert.ok(result.stdout.includes("lastAsyncUploadState=SUCCEEDED"), result.stdout);
});

test("a rejected status read fails closed with the store's explanation", async () => {
  const error = await runRelease(["status"], [() => rejectWith(400, IN_REVIEW_BODY)], { expectFailure: true });
  const output = error.stdout + error.stderr;
  assert.ok(output.includes("fetchStatus call returned HTTP 400"), output);
  assert.ok(output.includes("Item is currently in review and cannot be updated."), output);
  // This mode judges no version, so the message must not invent one.
  assert.ok(output.includes("v<unknown>"), output);
});

test("status names the fields the store sent, including shapes the gates do not read", async () => {
  const result = await runRelease(["status"], [
    () => ({
      itemId: extensionId,
      submittedItemRevisionStatus: {
        state: "STAGED",
        // `version`/`deployInfos` are the field names a reshaped response would
        // move this data into. Printing them as themselves is what tells a
        // release owner the gate asked the wrong question, rather than that the
        // rollout is still settling.
        distributionChannels: [
          { version: "0.1.12", deployInfos: [{ deployPercentage: 50 }] },
          { lastDeploy: "unrecognised" },
        ],
      },
    }),
  ]);
  assert.ok(result.stdout.includes('version=0.1.12 deployInfos=[{"deployPercentage":50}]'), result.stdout);
  // A channel carrying none of the known fields is echoed as itself instead of
  // being silently rendered as an empty entry.
  assert.ok(result.stdout.includes('<unrecognised shape: {"lastDeploy":"unrecognised"}>'), result.stdout);
});

test("status bounds the channel list instead of dumping it", async () => {
  // Mirrors STATUS_CHANNEL_LIMIT in chrome-web-store.sh: the summary is one line
  // by design, so it must not grow with the response.
  const channels = Array.from({ length: 6 }, (_, index) => ({
    crxVersion: `0.1.${index}`,
    deployPercentage: index,
  }));
  const result = await runRelease(["status"], [
    () => ({
      itemId: extensionId,
      submittedItemRevisionStatus: { state: "STAGED", distributionChannels: channels },
    }),
  ]);
  assert.ok(result.stdout.includes("crxVersion=0.1.3"), result.stdout);
  assert.ok(result.stdout.includes("| +2 more"), result.stdout);
  assert.ok(!result.stdout.includes("0.1.5"), `expected the list to be cut: ${result.stdout}`);
});

test("status bounds every value, not only the nested ones", async () => {
  // The first cut bounded the channel count and `deployInfos` but rendered state,
  // crxVersion and lastAsyncUploadState with a bare `tostring`, so a 3 MiB `state`
  // was measured at 3,145,904 bytes of run log -- the flood this summary exists to
  // prevent, arriving through the one field the first gate reads.
  const result = await runRelease(["status"], [
    () => ({
      itemId: extensionId,
      lastAsyncUploadState: "S".repeat(50_000),
      submittedItemRevisionStatus: {
        state: "R".repeat(3_000_000),
        distributionChannels: [{ crxVersion: "C".repeat(200_000), deployPercentage: 100 }],
      },
    }),
  ]);
  assert.ok(result.stdout.length < 4_000, `expected a bounded line, got ${result.stdout.length} characters`);
  assert.ok(result.stdout.includes("state=RRR"), result.stdout);
  assert.ok(!result.stdout.includes("R".repeat(200)), "the unbounded state reached the log");
  assert.ok(!result.stdout.includes("C".repeat(200)), "the unbounded crxVersion reached the log");
});

test("one unreadable shape cannot blank the rest of the summary", async () => {
  // Every case here aborted the single jq program and collapsed the whole line to
  // `<status response carried no readable fields>`, discarding the `state` the
  // first gate reads -- the one field the diagnosis cannot do without -- and, in
  // the middle case, a good entry standing beside the bad one.
  const cases = [
    {
      name: "channels that are not an array",
      status: { state: "STAGED", distributionChannels: "nonsense" },
      expect: ["submitted state=STAGED", '<not an array: "nonsense">'],
    },
    {
      name: "a scalar entry beside a good one",
      status: {
        state: "STAGED",
        distributionChannels: [{ crxVersion: VERSION, deployPercentage: 100 }, "junk"],
      },
      expect: [`crxVersion=${VERSION} deployPercentage=100`, '<unrenderable entry: "junk">'],
    },
    {
      name: "a revision that is not an object",
      status: "STAGED",
      expect: ["submitted state=<unrenderable state>"],
    },
  ];
  for (const { name, status, expect } of cases) {
    const result = await runRelease(["status"], [
      () => ({ itemId: extensionId, submittedItemRevisionStatus: status }),
    ]);
    assert.ok(result.stdout.includes("submitted"), `${name}: the revision went missing:\n${result.stdout}`);
    assert.ok(
      !result.stdout.includes("<status response carried no readable fields>"),
      `${name}: one bad shape blanked the whole summary:\n${result.stdout}`,
    );
    for (const fragment of expect) {
      assert.ok(result.stdout.includes(fragment), `${name}: expected ${JSON.stringify(fragment)} in:\n${result.stdout}`);
    }
  }
});

test("an absent revision and an empty channel list are reported differently", async () => {
  const result = await runRelease(["status"], [
    () => ({
      itemId: extensionId,
      submittedItemRevisionStatus: { state: "STAGED", distributionChannels: [] },
    }),
  ]);
  assert.ok(result.stdout.includes("submitted state=STAGED distributionChannels=[]"), result.stdout);
  // Not the same thing as a revision the store never sent.
  assert.ok(result.stdout.includes("published <absent>"), result.stdout);
});

test("the status summary redacts the access token, not only the error body", async () => {
  // `status` prints the summary and nothing else, so the body-only redaction left
  // this path able to echo the bearer token straight back through a field of the
  // response. Deleting the redaction from the summary kept the suite green before
  // this test existed.
  const token = "ya29.a0AfB_status-summary-secret-value";
  const result = await runRelease(["status"], [
    () => ({
      itemId: extensionId,
      submittedItemRevisionStatus: {
        state: `rejected request authorized by ${token}`,
        distributionChannels: [],
      },
    }),
  ], { token });
  const output = result.stdout + result.stderr;
  assert.ok(!output.includes(token), "the access token must never reach the log");
  // Redaction must not cost the diagnosis: the rest of the field still shows.
  assert.ok(output.includes("rejected request authorized by <redacted CWS_ACCESS_TOKEN>"), output);
});

test("status refuses arguments it cannot act on", async () => {
  // `status VERSION` reads like a check of that version; silently ignoring the
  // argument would let a caller believe it had been verified. No handler is
  // registered, so a run that asked the store anything would fail the request
  // count instead of passing quietly.
  await assert.rejects(
    runRelease(["status", VERSION], []),
    /status takes no arguments/,
  );
});

// The required-reviewers check was removed by operator decision on 2026-09-03
// (see verify-release-environment.sh). Every guard that REMAINS is asserted
// here one mutation at a time, because a suite that only covers the accept
// path plus one rejection stays green when a retained guard is deleted — and
// these guards are now the whole fail-closed contract, so a silent regression
// in one of them is exactly what would let an unprotected environment mint a
// release token. Each case perturbs a single field of the valid configuration.
const VALID_ENVIRONMENT = {
  deployment_branch_policy: { protected_branches: false, custom_branch_policies: true },
};
const VALID_BRANCHES = { total_count: 1, branch_policies: [{ name: "main", type: "branch" }] };

// The mock routes on the EXACT path and asserts the Authorization header. The
// previous catch-all matched any URL and ignored auth entirely, so it answered
// 200 to requests the real API refuses — which is precisely how the 403 that
// broke run 33793517588 stayed invisible to a green suite. `status` lets a case
// return a non-200 for one path so the failure path is exercised, not assumed.
const ENVIRONMENT_PATH = "/repos/owner/repository/environments/chrome-web-store";
const BRANCHES_PATH = `${ENVIRONMENT_PATH}/deployment-branch-policies`;

async function runVerifier({
  environment = VALID_ENVIRONMENT,
  branches = VALID_BRANCHES,
  status = {},
  token = "test-token",
  args = ["CWS_EXTENSION_ID", extensionId, "CWS_PUBLISHER_ID", "publisher"],
} = {}) {
  const requested = [];
  const server = createServer((request, response) => {
    const { pathname } = new URL(request.url, "http://127.0.0.1");
    requested.push(pathname);
    // A token-blind mock cannot catch an auth regression; the real API 401s.
    if (request.headers.authorization !== "Bearer test-token") {
      response.writeHead(401, { "Content-Type": "application/json" })
        .end(JSON.stringify({ message: "Bad credentials" }));
      return;
    }
    const bodies = { [ENVIRONMENT_PATH]: environment, [BRANCHES_PATH]: branches };
    if (!(pathname in bodies)) {
      // Unknown path is a hard failure rather than a default payload: the
      // verifier must not call an endpoint this harness has not modelled.
      response.writeHead(404, { "Content-Type": "application/json" })
        .end(JSON.stringify({ message: `unexpected path ${pathname}` }));
      return;
    }
    const code = status[pathname] ?? 200;
    if (code !== 200) {
      response.writeHead(code, {
        "Content-Type": "application/json",
        // Mirrors the header the real API returns, which the script now prints.
        "x-accepted-github-permissions": "environments=read",
      }).end(JSON.stringify({ message: "Resource not accessible by integration" }));
      return;
    }
    response.writeHead(200, { "Content-Type": "application/json" }).end(JSON.stringify(bodies[pathname]));
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  try {
    const result = await run("bash", [
      "scripts/verify-release-environment.sh", "chrome-web-store", ...args,
    ], {
      cwd: import.meta.dirname + "/..",
      env: {
        ...process.env,
        GITHUB_TOKEN: token,
        GITHUB_REPOSITORY: "owner/repository",
        GITHUB_API_URL: `http://127.0.0.1:${server.address().port}`,
      },
    });
    return { ...result, requested };
  } finally {
    await new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
  }
}

// Runs the mode that executes outside the environment. It makes no API call,
// so it needs no server and no token.
async function runUnscoped(pairs) {
  return run("bash", [
    "scripts/verify-release-environment.sh", "--assert-unscoped-empty", ...pairs,
  ], { cwd: import.meta.dirname + "/..", env: { ...process.env } });
}

test("protected environment verifier accepts the exact configuration", async () => {
  const result = await runVerifier();
  assert.match(result.stdout, /validated protected environment chrome-web-store/);
});

test("protected environment verifier requires a custom deployment branch policy", async () => {
  // Each case violates exactly ONE side of the `and`, so each condition is
  // pinned individually. The single fixture that used to stand here set
  // {protected: true, custom: false} — violating both at once, which either
  // half rejects on its own, so deleting either condition left the suite green
  // while the mutant accepted an environment with no branch restriction at all.
  await assert.rejects(
    runVerifier({ environment: { deployment_branch_policy: { protected_branches: true, custom_branch_policies: true } } }),
    /must use a custom deployment branch policy/,
  );
  await assert.rejects(
    runVerifier({ environment: { deployment_branch_policy: { protected_branches: false, custom_branch_policies: false } } }),
    /must use a custom deployment branch policy/,
  );
  // A null policy is GitHub's representation of "every branch may deploy" —
  // the most permissive configuration the API can report.
  await assert.rejects(
    runVerifier({ environment: { deployment_branch_policy: null } }),
    /must use a custom deployment branch policy/,
  );
});

test("protected environment verifier requires exactly the main branch", async () => {
  // Both halves matter: an extra allowed branch is a bypass, and a single
  // policy naming something other than main is a different bypass.
  await assert.rejects(
    runVerifier({
      branches: {
        total_count: 2,
        branch_policies: [{ name: "main", type: "branch" }, { name: "release/*", type: "branch" }],
      },
    }),
    /must allow exactly the main branch/,
  );
  await assert.rejects(
    runVerifier({ branches: { total_count: 1, branch_policies: [{ name: "develop", type: "branch" }] } }),
    /must allow exactly the main branch/,
  );
  // A TAG named `main` is the attack the `type` check exists to stop: tags are
  // not protected by the branch rules and anyone able to push one could deploy
  // arbitrary code. Without this case, deleting the `type` check left the suite
  // green while the mutant accepted it.
  await assert.rejects(
    runVerifier({ branches: { total_count: 1, branch_policies: [{ name: "main", type: "tag" }] } }),
    /must allow exactly the main branch/,
  );
  // `main*` is a glob, not the literal branch: it also matches `maintenance`,
  // `main-hotfix` and anything else a contributor can create.
  await assert.rejects(
    runVerifier({ branches: { total_count: 1, branch_policies: [{ name: "main*", type: "branch" }] } }),
    /must allow exactly the main branch/,
  );
});

test("protected environment verifier requires every variable to resolve inside the environment", async () => {
  // An empty value is what `vars.X` yields when X is not defined on the
  // environment at all, so this is the in-environment half of the scope check.
  await assert.rejects(
    runVerifier({ args: ["CWS_EXTENSION_ID", "", "CWS_PUBLISHER_ID", "publisher"] }),
    /CWS_EXTENSION_ID is empty or not defined on chrome-web-store/,
  );
});

test("protected environment verifier never calls the ungrantable variables endpoint", async () => {
  // GET environments/{env}/variables requires `environments=read`, which no
  // workflow token can hold (run 33793517588 died on it). The mock 404s any
  // unmodelled path, so a reintroduced call fails loudly here rather than in a
  // real release. Pinning the exact call list keeps that regression visible.
  const result = await runVerifier();
  assert.deepEqual(result.requested, [ENVIRONMENT_PATH, BRANCHES_PATH]);
  assert.ok(!result.requested.some((path) => path.endsWith("/variables")));
});

test("protected environment verifier fails closed on a 403 from the environment endpoint", async () => {
  // The exact failure of run 33793517588. It must abort the release, and the
  // message must name the status and the endpoint — the old `--fail-with-body`
  // path printed only "curl: (22) ... error: 403" with neither.
  await assert.rejects(
    runVerifier({ status: { [ENVIRONMENT_PATH]: 403 } }),
    (error) => {
      assert.match(error.stderr, /GitHub returned HTTP 403 for environments\/chrome-web-store/);
      assert.match(error.stderr, /x-accepted-github-permissions: environments=read/);
      assert.notEqual(error.code, 0);
      return true;
    },
  );
});

test("protected environment verifier fails closed on a 403 from the branch-policies endpoint", async () => {
  await assert.rejects(
    runVerifier({ status: { [BRANCHES_PATH]: 403 } }),
    /GitHub returned HTTP 403 for environments\/chrome-web-store\/deployment-branch-policies/,
  );
});

test("protected environment verifier fails closed when the token is rejected", async () => {
  // Must reach the mock's auth branch and get a real 401. Pointing this at a
  // dead port instead would fail on connection-refused and assert nothing about
  // authentication, while still passing — which is what it used to do.
  await assert.rejects(
    runVerifier({ token: "wrong-token" }),
    (error) => {
      assert.match(error.stderr, /GitHub returned HTTP 401 for environments\/chrome-web-store/);
      assert.match(error.stderr, /Bad credentials/);
      return true;
    },
  );
});

test("unscoped mode accepts variables that are invisible outside the environment", async () => {
  const result = await runUnscoped(["CWS_PUBLISHER_ID", "", "CWS_EXTENSION_ID", ""]);
  assert.match(result.stdout, /no release variable is defined at repository or organization scope/);
});

test("unscoped mode rejects a variable readable outside the environment", async () => {
  // A repository- or organization-scoped variable resolves non-empty in a job
  // with no `environment:` key. That is the misconfiguration this mode exists
  // to catch, and it is the half the API listing used to cover.
  await assert.rejects(
    runUnscoped(["CWS_PUBLISHER_ID", "", "CWS_EXTENSION_ID", extensionId]),
    /CWS_EXTENSION_ID is readable outside the environment/,
  );
});

test("unscoped mode rejects malformed argument pairs", async () => {
  await assert.rejects(
    runUnscoped(["CWS_PUBLISHER_ID"]),
    /expected NAME VALUE pairs/,
  );
});

// The variable-scope invariant no longer lives in a single API call that a
// script test can cover; it lives in the SHAPE of the two workflows. If the
// preflight job disappears, stops blocking the deploying job, gains an
// `environment:` key (which would make its variables resolve non-empty and the
// check vacuous), or checks out the dispatcher's ref instead of the reviewed
// one, the guard is gone while every other test here still passes. That is
// exactly what happened during review: deleting `needs: preflight`, or the
// whole preflight job, left the suite at 91/91 green.
//
// These parse the real workflow files. A dependency-free reader is used
// deliberately: the extension package ships no YAML parser, and adding one to
// devDependencies to assert four properties would be a heavier change to the
// store package's toolchain than the assertions are worth.
const STORE_WORKFLOWS = [
  { file: "chrome-web-store.yml", deployJob: "stage", environment: "chrome-web-store" },
  { file: "chrome-web-store-promote.yml", deployJob: "promote", environment: "chrome-web-store-production" },
];

// Reads the `jobs:` mapping into { jobName: { lines, keys } }. Only the
// structure these tests assert on is modelled; comments and blank lines are
// dropped so a comment mentioning `environment:` cannot be mistaken for the key.
function parseJobs(text) {
  const lines = text.split("\n").filter((line) => line.trim() !== "" && !/^\s*#/.test(line));
  const start = lines.findIndex((line) => line === "jobs:");
  assert.notEqual(start, -1, "workflow has no jobs: block");
  const jobs = {};
  let current = null;
  for (const line of lines.slice(start + 1)) {
    const header = /^ {2}([A-Za-z0-9_-]+):\s*$/.exec(line);
    if (header) {
      current = header[1];
      jobs[current] = { lines: [], keys: {} };
      continue;
    }
    if (!current) continue;
    jobs[current].lines.push(line);
    const key = /^ {4}([A-Za-z0-9_-]+):\s*(.*)$/.exec(line);
    if (key) jobs[current].keys[key[1]] = key[2].trim();
  }
  return jobs;
}

for (const { file, deployJob, environment } of STORE_WORKFLOWS) {
  test(`${file} keeps the preflight job that proves variables are environment-scoped`, async () => {
    const text = await readFile(new URL(`../../.github/workflows/${file}`, import.meta.url), "utf8");
    const jobs = parseJobs(text);

    // The job must exist at all — deleting it was one of the two silent kills.
    assert.ok(jobs.preflight, `${file} must define a preflight job`);

    // No `environment:` key. With one, `vars.*` would resolve non-empty inside
    // preflight and --assert-unscoped-empty would fail on a correct config, so
    // the only way to keep the pipeline green would be to weaken the check.
    assert.equal(
      jobs.preflight.keys.environment, undefined,
      `${file}: preflight must NOT declare an environment, or its variables resolve non-empty and the scope check proves nothing`,
    );

    // It must actually run the unscoped mode.
    assert.match(
      jobs.preflight.lines.join("\n"), /verify-release-environment\.sh --assert-unscoped-empty/,
      `${file}: preflight must run the verifier in --assert-unscoped-empty mode`,
    );

    // Preflight checks out the reviewed workflow ref. `ref: ${{ inputs.ref }}`
    // here would let a dispatch supply the verifier that judges it.
    assert.ok(
      !/ref:\s*\$\{\{\s*inputs\.ref/.test(jobs.preflight.lines.join("\n")),
      `${file}: preflight must check out the reviewed ref, never inputs.ref`,
    );
  });

  test(`${file} blocks ${deployJob} on preflight`, async () => {
    const text = await readFile(new URL(`../../.github/workflows/${file}`, import.meta.url), "utf8");
    const jobs = parseJobs(text);
    const deploy = jobs[deployJob];
    assert.ok(deploy, `${file} must define the ${deployJob} job`);

    // The hard dependency. Without it the two jobs run concurrently and a
    // failing preflight no longer stops the upload.
    assert.match(
      deploy.keys.needs ?? "", /\bpreflight\b/,
      `${file}: ${deployJob} must declare needs: preflight`,
    );
    assert.equal(deploy.keys.environment, environment);

    // `if:` or `continue-on-error:` would let the deploying job proceed past a
    // failed preflight, which is the same disarm by another route.
    assert.equal(
      deploy.keys.if, undefined,
      `${file}: ${deployJob} must not carry an if: condition that could bypass a failed preflight`,
    );
    for (const job of ["preflight", deployJob]) {
      assert.equal(
        jobs[job].keys["continue-on-error"], undefined,
        `${file}: ${job} must not set continue-on-error, which would make the gate advisory`,
      );
    }

    // The same disarm one level down. `continue-on-error:`/`if:` on preflight's
    // STEP, or a `|| true` appended to the verifier call, leaves preflight
    // reporting success on a real leak while `needs: preflight` still looks
    // intact — so check every indent inside the job, not just its job keys.
    const preflightBody = jobs.preflight.lines.join("\n");
    assert.ok(
      !/^\s+continue-on-error:/m.test(preflightBody),
      `${file}: no step in preflight may set continue-on-error`,
    );
    assert.ok(
      !/^\s+if:/m.test(preflightBody),
      `${file}: no step in preflight may carry an if: condition`,
    );
    // A trailing `|| true` / `|| :` swallows the verifier's non-zero exit.
    assert.ok(
      !/verify-release-environment\.sh[^\n]*(\|\||;)/.test(preflightBody.replace(/\\\n/g, " ")),
      `${file}: preflight's verifier call must not be chained with || or ;, which would swallow its exit code`,
    );
  });

  test(`${file} checks the same variables in preflight and in the environment`, async () => {
    // F4: the names are listed twice per workflow with nothing keeping them in
    // sync, so a fifth variable added only to the deploying job would never be
    // scope-checked. Compare the two lists rather than trusting review.
    //
    // Match the ARGUMENT list (`NAME "$NAME"` pairs in the `run:` block), not
    // the `env:` mapping: the script only ever checks the names passed to it
    // positionally, so an `env:` entry with no matching argument is not
    // verified at all. Pinning `env:` instead left three disarms green —
    // truncating preflight's arguments, passing `""` for all four, and adding
    // a fifth variable to both `env:` blocks but only one argument list.
    const text = await readFile(new URL(`../../.github/workflows/${file}`, import.meta.url), "utf8");
    const jobs = parseJobs(text);
    const names = (job) => [...new Set(
      [...job.lines.join("\n").matchAll(/^\s+(CWS_[A-Z_]+|GCP_[A-Z_]+) "\$\{?(?:CWS_[A-Z_]+|GCP_[A-Z_]+)\}?"/gm)].map((m) => m[1]),
    )].sort();
    const preflightNames = names(jobs.preflight);
    assert.deepEqual(
      preflightNames, names(jobs[deployJob]),
      `${file}: preflight and ${deployJob} must check the same variable names`,
    );
    assert.ok(preflightNames.length >= 4, "expected at least the four release variables");
  });
}
