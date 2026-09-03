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

async function runRelease(args, handlers) {
  let requestIndex = 0;
  const server = createServer(async (request, response) => {
    const chunks = [];
    for await (const chunk of request) chunks.push(chunk);
    try {
      assert.equal(request.headers.authorization, "Bearer test-token");
      const handler = handlers[requestIndex++];
      assert.ok(handler, `unexpected request ${request.method} ${request.url}`);
      const payload = handler(request, Buffer.concat(chunks));
      response.writeHead(200, { "Content-Type": "application/json" });
      response.end(JSON.stringify(payload));
    } catch (error) {
      response.writeHead(500, { "Content-Type": "application/json" });
      response.end(JSON.stringify({ error: String(error) }));
    }
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address();
  try {
    const result = await run("bash", ["scripts/chrome-web-store.sh", ...args], {
      cwd: import.meta.dirname + "/..",
      env: {
        ...process.env,
        CWS_API_ROOT: `http://127.0.0.1:${port}`,
        CWS_ACCESS_TOKEN: "test-token",
        CWS_PUBLISHER_ID: "test-publisher",
        CWS_EXTENSION_ID: extensionId,
        CWS_POLL_INTERVAL_SECONDS: "0",
      },
    });
    assert.equal(requestIndex, handlers.length);
    return result;
  } finally {
    await new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
  }
}

test("manifest requests tabGroups exactly once", async () => {
  const manifest = JSON.parse(await readFile(new URL("../manifest.json", import.meta.url), "utf8"));
  assert.equal(manifest.version, "0.1.7");
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
const VALID_VARIABLES = {
  variables: [
    { name: "CWS_EXTENSION_ID", value: extensionId },
    { name: "CWS_PUBLISHER_ID", value: "publisher" },
  ],
};

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
        GITHUB_TOKEN: "test-token",
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
  await assert.rejects(
    runVerifier({ environment: { deployment_branch_policy: { protected_branches: true, custom_branch_policies: false } } }),
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
});

test("protected environment verifier requires every variable to resolve inside the environment", async () => {
  // An empty value is what `vars.X` yields when X is not defined on the
  // environment at all, so this is the in-environment half of the scope check.
  await assert.rejects(
    runVerifier({ args: ["CWS_EXTENSION_ID", "", "CWS_PUBLISHER_ID", "publisher"] }),
    /CWS_EXTENSION_ID is not defined on chrome-web-store/,
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
  // The harness now checks Authorization, so a wrong token yields a real 401
  // instead of the catch-all 200 the previous mock returned.
  await assert.rejects(
    run("bash", ["scripts/verify-release-environment.sh", "chrome-web-store", "CWS_PUBLISHER_ID", "publisher"], {
      cwd: import.meta.dirname + "/..",
      env: { ...process.env, GITHUB_TOKEN: "wrong-token", GITHUB_REPOSITORY: "owner/repository", GITHUB_API_URL: "http://127.0.0.1:1" },
    }),
    /release environment validation failed/,
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
