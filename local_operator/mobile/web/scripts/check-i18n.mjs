/** Classic TypeScript (5.8+ compiler API) copy gate, independent of desktop TS7.
 * Exact literal+sink MULTISETS prevent replacement copy hiding at equal counts.
 * Activation is reviewed path scope; old application copy remains advisory. */
import ts from "typescript";
import { readFileSync, readdirSync } from "node:fs";
import { dirname, join, relative } from "node:path";
import { fileURLToPath } from "node:url";

export const ENFORCED = ["src/i18n/runtime.ts", "src/i18n/wire-cache.ts", "src/store.ts", "src/main.tsx", "src/screens/session-list.tsx", "src/screens/session-view.tsx", "src/screens/past-sessions.tsx", "src/components/directory-sheet.tsx", "src/components/model-sheet.tsx"];
export const NONTRANSLATABLE = [{
	path: "src/i18n/wire-cache.ts", sink: "new:Error", text: "catalogue response limit",
	reason: "Internal payload bound caught by loader; never surfaced as user copy.",
}];
const copyName = /(?:^|[._])(text|message|label|help|title|description|placeholder|summary|errorText)$/i;
const attributes = new Set(["title", "alt", "placeholder", "aria-label", "aria-description", "aria-valuetext"]);
const prose = value => /\p{L}/u.test(value);
// Checkout-independent comparison: a Windows checkout (CRLF, autocrlf) yields
// literal text whose indentation/newlines are \r\n, while frozen baseline keys
// hold bare \n, so the same source would fingerprint differently. Normalising to
// \n on both sides removes the asymmetry without exempting any file or weakening
// the exact multiset semantics.
const normalize = value => value.replace(/\r\n?/g, "\n");
export function scanSource(source, path) {
	source = normalize(source);
	const tree = ts.createSourceFile(path, source, ts.ScriptTarget.Latest, true, path.endsWith(".tsx") ? ts.ScriptKind.TSX : ts.ScriptKind.TS);
	if (tree.parseDiagnostics.length) throw new Error(`Cannot scan invalid source: ${path}`);
	const found = [];
	function literal(node, sink) {
		if (!node) return;
		if (ts.isStringLiteralLike(node) || ts.isJsxText(node)) {
			const text = normalize(node.text.trim());
			if (prose(text) && !NONTRANSLATABLE.some(item => item.path === path && item.sink === sink && item.text === text)) found.push({ sink, text });
			return;
		}
		if (ts.isTemplateExpression(node)) {
			literal(node.head, sink);
			for (const span of node.templateSpans) { literal(span.expression, sink); literal(span.literal, sink); }
			return;
		}
		if (ts.isTemplateHead(node) || ts.isTemplateMiddle(node) || ts.isTemplateTail(node)) {
			const text = normalize(node.text.trim()); if (prose(text)) found.push({ sink, text }); return;
		}
		if (ts.isConditionalExpression(node)) { literal(node.whenTrue, sink); literal(node.whenFalse, sink); }
		else if (ts.isBinaryExpression(node)) { literal(node.left, sink); literal(node.right, sink); }
		else if (ts.isParenthesizedExpression(node)) literal(node.expression, sink);
		else if (ts.isJsxExpression(node)) literal(node.expression, sink);
		else if (ts.isTypeNode(node)) return;
		else if (ts.isJsxAttribute(node)) { if (attributes.has(node.name.getText(tree))) literal(node.initializer, sink); }
		else if (ts.isJsxSpreadAttribute(node)) {
			if (ts.isObjectLiteralExpression(node.expression)) for (const property of node.expression.properties) {
				if (ts.isPropertyAssignment(property) && attributes.has(property.name.getText(tree).replace(/^["']|["']$/g, ""))) literal(property.initializer, sink);
			}
		}
		else if (ts.isPropertyAssignment(node)) literal(node.initializer, sink);
		else if (ts.isPropertyAccessExpression(node) || ts.isElementAccessExpression(node)) literal(node.expression, sink);
		else if (ts.isCallExpression(node)) {
			literal(node.expression, sink);
			for (const [index, arg] of node.arguments.entries()) {
				// A generated translator key is a lookup token, not rendered copy.
				// This position-specific rule does not exempt arbitrary strings.
				if (index === 0 && /(?:^|\.)t$/.test(node.expression.getText(tree)) && ts.isStringLiteralLike(arg) && /^(?:relay|wire)(?:\.[a-z][a-z0-9_]*)+$/.test(arg.text)) continue;
				literal(arg, sink);
			}
		}
		else ts.forEachChild(node, child => literal(child, sink));
	}
	function visit(node) {
		if (ts.isJsxText(node)) literal(node, "jsx:text");
		else if (ts.isJsxAttribute(node) && attributes.has(node.name.getText(tree))) literal(node.initializer, `jsx:${node.name.getText(tree)}`);
		else if (ts.isJsxExpression(node) && !ts.isJsxAttribute(node.parent)) literal(node.expression, "jsx:expression");
		else if (ts.isVariableDeclaration(node) && copyName.test(node.name.getText(tree))) literal(node.initializer, `variable:${node.name.getText(tree)}`);
		else if (ts.isPropertyAssignment(node) && copyName.test(node.name.getText(tree).replace(/^["']|["']$/g, ""))) literal(node.initializer, `property:${node.name.getText(tree)}`);
		else if (ts.isBinaryExpression(node) && [ts.SyntaxKind.EqualsToken, ts.SyntaxKind.PlusEqualsToken].includes(node.operatorToken.kind)) {
			const target = node.left;
			const name = ts.isPropertyAccessExpression(target) ? target.name.text : ts.isElementAccessExpression(target) && ts.isStringLiteralLike(target.argumentExpression) ? target.argumentExpression.text : null;
			// These are rendered copy sinks, unlike technical lang/dir/class or URL
			// assignments. A machine-only title still needs a precise reviewed rule.
			if (name && ["title", "textContent", "innerText", "innerHTML", "placeholder", "alt", "ariaLabel", "ariaDescription"].includes(name)) literal(node.right, `assignment:${name}`);
		}
		else if (ts.isReturnStatement(node)) {
			let owner = node.parent;
			while (owner && !ts.isFunctionLike(owner)) owner = owner.parent;
			const name = owner?.name?.getText(tree) ?? (owner && ts.isVariableDeclaration(owner.parent) ? owner.parent.name.getText(tree) : "");
			if (["t", "render", "renderEnvelope"].includes(name)) literal(node.expression, `return:${name}`);
		}
		else if (ts.isNewExpression(node) && node.expression.getText(tree) === "Error") for (const arg of node.arguments ?? []) literal(arg, "new:Error");
		else if (ts.isCallExpression(node)) {
			const name = node.expression.getText(tree);
			if (name.endsWith(".setAttribute") && node.arguments.length >= 2 && ts.isStringLiteralLike(node.arguments[0]) && attributes.has(node.arguments[0].text)) literal(node.arguments[1], `attribute:${node.arguments[0].text}`);
			if (/(?:set(?:Error|Notice|Message|Label)|toast|alert|confirm|\.push)$/.test(name)) for (const arg of node.arguments) literal(arg, `call:${name}`);
		}
		ts.forEachChild(node, visit);
	}
	visit(tree); return found;
}
export function fingerprints(found) {
	const counts = {};
	for (const entry of found) { const key = JSON.stringify([entry.sink, entry.text]); counts[key] = (counts[key] ?? 0) + 1; }
	return counts;
}
function files(directory) {
	return readdirSync(directory, { withFileTypes: true }).flatMap(item => item.isDirectory() ? files(join(directory, item.name)) : /\.tsx?$/.test(item.name) && !/\.(?:test|type-test)\.tsx?$/.test(item.name) ? [join(directory, item.name)] : []);
}
export function checkTree(root, baseline) {
	if (baseline.schema !== 1 || JSON.stringify(baseline.enforced) !== JSON.stringify(ENFORCED)) throw new Error("Invalid reviewed i18n enforcement scope");
	const failures = []; let advisory = 0; const seen = new Set();
	for (const path of files(join(root, "src"))) {
		const name = relative(root, path).replaceAll("\\", "/");
		seen.add(name);
		const actual = fingerprints(scanSource(readFileSync(path, "utf8"), name));
		if (!ENFORCED.includes(name)) { advisory += Object.values(actual).reduce((a, b) => a + b, 0); continue; }
		const prior = baseline.legacy[name] ?? {};
		for (const fingerprint of new Set([...Object.keys(actual), ...Object.keys(prior)])) {
			const count = actual[fingerprint] ?? 0;
			if (count !== (prior[fingerprint] ?? 0)) failures.push({ path: name, fingerprint, count, allowed: prior[fingerprint] ?? 0 });
		}
	}
	for (const path of ENFORCED) if (!seen.has(path)) failures.push({ path, fingerprint: "missing activated source", count: 0, allowed: 1 });
	return { failures, advisory };
}
export function main(argv = process.argv.slice(2)) {
	const index = argv.indexOf("--root");
	const root = index < 0 ? join(dirname(fileURLToPath(import.meta.url)), "..") : argv[index + 1];
	try {
		const baseline = JSON.parse(readFileSync(join(root, "i18n-baseline.json"), "utf8"));
		const result = checkTree(root, baseline);
		for (const finding of result.failures) console.error(JSON.stringify(finding));
		console.log(`relay i18n: ${result.failures.length} enforced fingerprint violation(s); ${result.advisory} advisory legacy sink literal(s)`);
		return result.failures.length ? 1 : 0;
	} catch (error) { console.error(String(error)); return 1; }
}
if (process.argv[1] === fileURLToPath(import.meta.url)) process.exitCode = main();
