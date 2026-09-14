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
 * The second test is the closure half of the same invariant, and it is the half
 * a `chrome`-only scan misses: importing `../state` from a driver module drags
 * the chrome-coupled storage layer into the vendored copy, with no chrome
 * reference anywhere in this directory for the first test to find.
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
  if (ts.isPropertyAccessExpression(parent) && parent.name === node) return true;
  if (ts.isQualifiedName(parent) && parent.right === node) return true;
  if (
    (ts.isPropertyAssignment(parent) ||
      ts.isShorthandPropertyAssignment(parent) ||
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

/** The `chrome`-referencing nodes of one parsed driver module. */
function chromeReferences(sf) {
  const hits = [];
  const visit = (node) => {
    if (ts.isIdentifier(node) && node.text === "chrome" && !isNamePosition(node)) {
      hits.push({ pos: node.getStart(sf), what: "the `chrome` global" });
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

/** Module specifiers of every import/export-from in a driver module. */
function importedSpecifiers(sf) {
  const found = [];
  const visit = (node) => {
    if (
      (ts.isImportDeclaration(node) || ts.isExportDeclaration(node)) &&
      node.moduleSpecifier &&
      ts.isStringLiteral(node.moduleSpecifier)
    ) {
      found.push(node.moduleSpecifier.text);
    }
    ts.forEachChild(node, visit);
  };
  visit(sf);
  return found;
}

test("no module under src/driver/ references the chrome global", async () => {
  const modules = await walk(DRIVER);
  for (const expected of EXPECTED_MODULES) {
    assert.ok(
      modules.includes(expected),
      `src/driver/${expected} is missing: the host-free set moved, so this guard is now checking the wrong tree`,
    );
  }

  const references = [];
  for (const rel of modules) {
    const source = await readFile(new URL(rel, DRIVER), "utf8");
    references.push(...chromeReferences(parse(rel, source)));
  }
  assert.deepEqual(
    references,
    [],
    "src/driver/ is the host-free set a second host vendors: nothing here may reference chrome",
  );
});

test("no module under src/driver/ imports its way back out to a host-coupled module", async () => {
  const modules = await walk(DRIVER);
  const offenders = [];
  for (const rel of modules) {
    const source = await readFile(new URL(rel, DRIVER), "utf8");
    for (const specifier of importedSpecifiers(parse(rel, source))) {
      if (!specifier.startsWith(".")) continue; // bare specifiers are not this repo's modules
      const resolved = new URL(specifier, new URL(rel, DRIVER));
      const inside = resolved.pathname.startsWith(DRIVER.pathname);
      if (!inside && !ALLOWED_EXTERNAL_IMPORTS.has(specifier)) {
        offenders.push(`src/driver/${rel} imports ${specifier}`);
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
      if (!exists) offenders.push(`src/driver/${rel} imports ${specifier}, which resolves to no file`);
    }
  }
  assert.deepEqual(
    offenders,
    [],
    "a driver module may only import existing modules from src/driver/ (or protocol.gen.ts): an outside import drags the chrome-coupled layer into the vendored copy",
  );
});
