/* The invariant the cross-repo sharing arrangement rests on: every module under
 * `src/driver/` is HOST-FREE.
 *
 * WHY IT IS A TEST AND NOT A REVIEW HABIT. `driver/` is the set of modules a
 * second host (the desktop app's Electron browser tab) vendors whole, and
 * `scripts/`-generated copies of it are taken from this directory at a pinned
 * ref. The property that makes that possible — no `chrome.*` call, no type from
 * `@types/chrome`, and no import that would drag either in — is invisible in
 * the diff of the change that could break it: one `chrome.storage` call added
 * to a helper here compiles, typechecks, and passes every other suite in this
 * repo, and only fails in a host that has no `chrome` global at all. So the
 * check is mechanical.
 *
 * WHY THE TYPESCRIPT PARSER RATHER THAN A REGEX. Two reasons, and both have
 * bitten this tree:
 *   1. These modules' rationale comments deliberately NAME chrome APIs (the
 *      ceiling table explains what `chrome.debugger.sendCommand` costs, and
 *      `errors.ts` explains the `chrome.debugger.onDetach` cycle it breaks). A
 *      text search for `chrome.` flags those comments, so it would either fail
 *      on correct code or have to strip comments — and stripping them by regex
 *      corrupts string literals (`"https://example.test/"` loses everything
 *      from `//` on), which is exactly how a real reference would hide.
 *   2. `chrome` can be reached without any member access at all: `typeof chrome`,
 *      a `chrome` type reference, or `globalThis["chrome"]`. The AST sees all of
 *      them; `chrome.` as a literal does not.
 * A comment or a doc string that mentions `chrome.` is not a dependency and is
 * allowed. An identifier that resolves to the global is not.
 *
 * WHAT COUNTS AS A REFERENCE, exhaustively — this list is the claim, and an
 * independent review of the first version found it overstated (design §12.2):
 *   1. a bare identifier in value position: `chrome`, `typeof chrome`;
 *   2. a global object's property: `globalThis.chrome`, `window.chrome`,
 *      `self.chrome`, `global.chrome` (the dot form — the identifier is the
 *      property NAME, which is why a name-position scan misses it);
 *   3. an element access whose key is the string: `globalThis["chrome"]`;
 *   4. the shorthand `{ chrome }`, which is a value reference to whatever
 *      binding holds that name, never a property being declared;
 *   5. a `chrome` type reference (`chrome.tabs.Tab` names the GLOBAL namespace).
 * Position (4) and the `.chrome` of `obj.chrome` (an unrelated object's own
 * field) are the two cases that need opposite treatment, and they are the two a
 * plain AST walk gets wrong in opposite directions.
 *
 * The second test is the closure half of the same invariant, and it is the half
 * a `chrome`-only scan misses: importing `../state` from a driver module drags
 * the chrome-coupled storage layer into the vendored copy, with no chrome
 * reference anywhere in this directory for the first test to find. It covers
 * DYNAMIC forms too — `await import("../state")` and `require("../cdp")` — and
 * it fails CLOSED on a computed specifier it cannot resolve, because "this
 * import cannot be checked" is not "this import is fine".
 *
 * THE SCOPE IS EVERYTHING WE VENDOR, not just this directory: the generated
 * `src/protocol.gen` that driver modules are allowed to import (an allow-list
 * entry whose host-freedom was previously asserted and never checked) and the
 * bundle generated for the second host (`extension/ui-vendor/`, produced by
 * `python -m local_operator.browser_bridge.gen_ts`). Both ship across the repo
 * boundary, so both are held to the same rule.
 */
import assert from "node:assert/strict";
import test from "node:test";
import { readdir, readFile, stat } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import ts from "typescript";

const DRIVER = new URL("../src/driver/", import.meta.url);

/** Every module the sharing arrangement vendors. Pinned so the walk cannot pass
 * vacuously: a missing file here means the directory moved and the guard below
 * is checking nothing, which is worse than not having it. Extra modules are
 * allowed and are covered automatically — the walk is the scope. */
const EXPECTED_MODULES = [
  "access-flow.ts",
  "access-queue.ts",
  "ax-compact.ts",
  "deadline.ts",
  "errors.ts",
  "origin-policy.ts",
  "psl.gen.ts",
  "scroll-expressions.ts",
];

/** Imports a driver module may have that do NOT resolve inside `driver/`.
 *
 * `protocol.gen.ts` is the generated wire vocabulary (`PROTO_VERSION`,
 * `ErrorCode`, the method union). It is host-free and vendored too, under its
 * own path, so a driver module may depend on it — but it is generated, so it is
 * not itself a driver module. Everything else a driver module imports must be
 * inside this directory: an import of `../state` or `./cdp` would pull the
 * chrome-coupled layer in behind a directory that reads as clean. */
const ALLOWED_EXTERNAL_IMPORTS = new Set(["../protocol.gen"]);

/** Objects whose `chrome` property IS the global, whichever spelling reaches it.
 * `obj.chrome` on anything else is that object's own field and is allowed — the
 * distinction is the whole reason this list exists rather than a rule like
 * "any property access named chrome". */
const GLOBAL_OBJECTS = new Set(["globalThis", "window", "self", "global"]);

/** The bundle this repository generates for the second host, relative to
 * `extension/`. Scanned as VENDORED, because it is the copy that ships. */
const VENDORED_BUNDLE = "../ui-vendor/";

/** The generated wire vocabulary driver modules may import, relative to
 * `extension/`. Scanned because the allow-list grants an exemption, and an
 * exemption that is never checked is how the exemption stops being true. */
const VENDORED_PROTOCOL = "../src/protocol.gen.ts";

async function walk(dirUrl, prefix = "") {
  const found = [];
  for (const entry of await readdir(dirUrl, { withFileTypes: true })) {
    const rel = `${prefix}${entry.name}`;
    if (entry.isDirectory()) {
      found.push(...(await walk(new URL(`${entry.name}/`, dirUrl), `${rel}/`)));
    } else if (entry.name.endsWith(".ts")) {
      found.push(rel);
    }
  }
  return found.sort();
}

function parse(rel, text) {
  return ts.createSourceFile(rel, text, ts.ScriptTarget.ES2022, true, ts.ScriptKind.TS);
}

/** Positions that hold a NAME rather than a value: `obj.chrome`, a
 * `chrome:` property key, the `tabs` of the qualified type `chrome.tabs.Tab`.
 * Everything else named `chrome` is a reference to the global. */
function isNamePosition(node) {
  const parent = node.parent;
  if (!parent) return false;
  // `globalThis.chrome` and friends: the NAME slot of a GLOBAL OBJECT is a
  // reference, so it is not exempt. Every other `obj.chrome` is that object's
  // own field and is.
  if (ts.isPropertyAccessExpression(parent) && parent.name === node) {
    return !isGlobalObjectAccess(node);
  }
  if (ts.isQualifiedName(parent) && parent.right === node) return true;
  // Deliberately NOT `isShorthandPropertyAssignment`: `{ chrome }` is a value
  // read of whatever binding holds that name, not a property being declared.
  // The declaration forms below have a name that is never a read.
  if (
    (ts.isPropertyAssignment(parent) ||
      ts.isPropertySignature(parent) ||
      ts.isPropertyDeclaration(parent) ||
      ts.isMethodDeclaration(parent) ||
      ts.isMethodSignature(parent)) &&
    parent.name === node
  ) {
    return true;
  }
  return false;
}

/** `globalThis.chrome` / `window.chrome` / `self.chrome` / `global.chrome`:
 * the property NAME slot of a global object, which is a reference to the global
 * and the one spelling a name-position exemption silently swallows. */
function isGlobalObjectAccess(node) {
  const parent = node.parent;
  if (!ts.isPropertyAccessExpression(parent) || parent.name !== node) return false;
  const target = parent.expression;
  return ts.isIdentifier(target) && GLOBAL_OBJECTS.has(target.text);
}

/** `globalThis["chrome"]` / `chrome["tabs"]` — an element access is the one
 * route to the global that has no identifier to find. */
function isChromeStringIndex(node) {
  return (
    ts.isStringLiteral(node) &&
    node.text === "chrome" &&
    node.parent &&
    ts.isElementAccessExpression(node.parent) &&
    node.parent.argumentExpression === node
  );
}

/** The `chrome`-referencing nodes of one parsed vendored module. */
function chromeReferences(sf) {
  const hits = [];
  const visit = (node) => {
    if (ts.isIdentifier(node) && node.text === "chrome" && !isNamePosition(node)) {
      hits.push({
        pos: node.getStart(sf),
        what: isGlobalObjectAccess(node)
          ? "the `chrome` global, through a global object's property"
          : "the `chrome` global",
      });
    } else if (isChromeStringIndex(node)) {
      hits.push({ pos: node.getStart(sf), what: 'the string key "chrome"' });
    }
    ts.forEachChild(node, visit);
  };
  visit(sf);
  return hits.map((hit) => {
    const { line, character } = sf.getLineAndCharacterOfPosition(hit.pos);
    return `${sf.fileName}:${line + 1}:${character + 1} references ${hit.what}`;
  });
}

/** Every STATIC and DYNAMIC module reference of one parsed vendored module.
 *
 * Three forms, because a chrome-coupled dependency does not have to be named in
 * an `import` statement: a static `import`/`export ... from`, `await import(...)`
 * and `require(...)`. The first version of this guard saw only the first, and a
 * dynamic import of `../state` therefore left both of its tests green.
 *
 * A reference whose specifier is not a string literal is returned with
 * `specifier: null` so the caller can fail it CLOSED: the guard's claim is
 * "nothing here can reach chrome", and a specifier nobody can read cannot
 * support that claim.
 */
function moduleReferences(sf) {
  const found = [];
  const literal = (node) => {
    if (!node) return null;
    if (ts.isStringLiteral(node) || ts.isNoSubstitutionTemplateLiteral(node)) return node.text;
    return null;
  };
  const visit = (node) => {
    if (
      (ts.isImportDeclaration(node) || ts.isExportDeclaration(node)) &&
      node.moduleSpecifier
    ) {
      found.push({
        kind: "import",
        specifier: literal(node.moduleSpecifier),
        pos: node.getStart(sf),
      });
    } else if (ts.isCallExpression(node)) {
      const callee = node.expression;
      const dynamic = callee.kind === ts.SyntaxKind.ImportKeyword;
      const required = ts.isIdentifier(callee) && callee.text === "require";
      if (dynamic || required) {
        found.push({
          kind: dynamic ? "dynamic import()" : "require()",
          specifier: literal(node.arguments[0]),
          pos: node.getStart(sf),
        });
      }
    }
    ts.forEachChild(node, visit);
  };
  visit(sf);
  return found;
}

/** Every module this repository VENDORS, as `{ label, url, root, allowExternal }`.
 *
 * Three scopes, because "host-free" is a property of what crosses the repo
 * boundary, not of one directory:
 *   * `src/driver/**` — the shared set, whose imports may reach `../protocol.gen`
 *     and nothing else outside itself;
 *   * `src/protocol.gen.ts` — the exemption driver modules rely on, which must
 *     therefore be checked rather than assumed, and which may import NOTHING
 *     local (the consumer copies it alone);
 *   * `ui-vendor/**` — the bundle generated for the second host, which is the
 *     copy that actually ships, so the same rule applies to it verbatim.
 */
async function vendoredModules() {
  const entries = [];
  for (const rel of await walk(DRIVER)) {
    entries.push({
      label: `src/driver/${rel}`,
      url: new URL(rel, DRIVER),
      root: DRIVER,
      allowExternal: ALLOWED_EXTERNAL_IMPORTS,
    });
  }
  const bundle = new URL(VENDORED_BUNDLE, import.meta.url);
  assert.ok(
    await stat(fileURLToPath(bundle)).then(
      () => true,
      () => false,
    ),
    `${VENDORED_BUNDLE} is missing: it is the copy the second host vendors, and without it this guard is silent about the artefact that ships`,
  );
  for (const rel of await walk(bundle)) {
    entries.push({
      label: `ui-vendor/${rel}`,
      url: new URL(rel, bundle),
      root: bundle,
      allowExternal: new Set(),
    });
  }
  entries.push({
    label: "src/protocol.gen.ts",
    url: new URL(VENDORED_PROTOCOL, import.meta.url),
    root: new URL(VENDORED_PROTOCOL, import.meta.url),
    allowExternal: new Set(),
  });
  return entries.sort((a, b) => a.label.localeCompare(b.label));
}

test("no vendored module references the chrome global", async () => {
  const references = [];
  for (const { label, url } of await vendoredModules()) {
    references.push(...chromeReferences(parse(label, await readFile(url, "utf8"))));
  }
  assert.deepEqual(
    references,
    [],
    "everything under src/driver/, the generated src/protocol.gen.ts and the ui-vendor/ bundle is what a second host vendors whole: nothing there may reference chrome",
  );
});

test("the guard's scope is the whole vendored set, and it is not empty", async () => {
  // A walk that finds nothing passes both tests above while checking nothing,
  // which is worse than not having the guard. So the scope is asserted.
  const modules = await walk(DRIVER);
  for (const expected of EXPECTED_MODULES) {
    assert.ok(
      modules.includes(expected),
      `src/driver/${expected} is missing: the host-free set moved, so this guard is now checking the wrong tree`,
    );
  }
  const bundle = await walk(new URL(VENDORED_BUNDLE, import.meta.url));
  assert.ok(
    bundle.includes("protocol.gen.ts"),
    "the vendored bundle carries no protocol.gen.ts: run `python -m local_operator.browser_bridge.gen_ts`",
  );
  assert.ok(
    bundle.some((rel) => rel.startsWith("driver/")),
    "the vendored bundle carries no driver/ modules: it is generated from extension/src/driver/, so regenerate it after any change there (`python -m local_operator.browser_bridge.gen_ts`)",
  );
  const entries = await vendoredModules();
  const seen = new Set(entries.map((entry) => entry.label));
  for (const expected of EXPECTED_MODULES) {
    assert.ok(seen.has(`src/driver/${expected}`), `${expected} is not being scanned`);
    assert.ok(seen.has(`ui-vendor/driver/${expected}`), `the vendored copy of ${expected} is not being scanned`);
  }
  assert.ok(seen.has("src/protocol.gen.ts"), "the allow-listed generated protocol is not being scanned");
});

test("no vendored module imports its way back out to a host-coupled module", async () => {
  const offenders = [];
  for (const { label, url, root, allowExternal } of await vendoredModules()) {
    const source = await readFile(url, "utf8");
    for (const reference of moduleReferences(parse(label, source))) {
      if (reference.specifier === null) {
        offenders.push(
          `${label} has a ${reference.kind} whose specifier is not a string literal: the vendored copy has no resolver, so this guard cannot support its claim about it`,
        );
        continue;
      }
      const specifier = reference.specifier;
      if (!specifier.startsWith(".")) continue; // bare specifiers are not this repo's modules
      const resolved = new URL(specifier, url);
      const inside = resolved.pathname.startsWith(root.pathname);
      if (!inside && !allowExternal.has(specifier)) {
        offenders.push(`${label} imports ${specifier} (via ${reference.kind})`);
        continue;
      }
      // The target must EXIST. A driver module that imports `./state` resolves
      // outside `driver/` and is caught above, but one that imports a sibling
      // which does not exist (a type left behind in the old location, say) is
      // only visible as "does this file resolve at all" — and that failure is
      // otherwise reported by `tsc` on whichever host compiles the tree, which
      // is exactly the feedback the vendored copy does not get.
      const target = `${fileURLToPath(resolved)}.ts`;
      const exists = await stat(target).then(
        () => true,
        () => false,
      );
      if (!exists) {
        offenders.push(
          `${label} imports ${specifier} (via ${reference.kind}), which resolves to no file`,
        );
      }
    }
  }
  assert.deepEqual(
    offenders,
    [],
    "a vendored module may only import existing modules from its own vendored tree (a driver module may also import protocol.gen.ts): an outside import drags the chrome-coupled layer into the copy that ships, and a dynamic one does it without a specifier a static scan can see",
  );
});
