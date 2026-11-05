import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

// Reuse the already-installed ICU parser; no package install or application build.
const require = createRequire(`${process.env.HOME}/local-operator-ui/package.json`);
const { parse, TYPE } = require('@formatjs/icu-messageformat-parser');
const root = path.dirname(fileURLToPath(import.meta.url));
const load = name => JSON.parse(readFileSync(path.join(root, name), 'utf8'));
const en = load('inputs/en.json');
const context = load('inputs/en.context.json');
const fr = load('wire.settings.account_policy.json');
const bidi = /[\u061c\u200e\u200f\u202a-\u202e\u2066-\u2069]/u;
function validate(target) {
  assert.deepEqual(Object.keys(target), Object.keys(en), 'exact key parity and source order');
  for (const [key, text] of Object.entries(target)) {
    assert.equal(typeof text, 'string');
    assert(text.trim().length > 0, `${key}: nonempty`);
    assert.notEqual(text, en[key], `${key}: not copied English`);
    assert(!bidi.test(text), `${key}: no manual bidi controls`);
    // All source messages are argument-free. Require target AST to remain literal.
    assert(parse(en[key]).every(node => node.type === TYPE.literal));
    assert(parse(text).every(node => node.type === TYPE.literal), `${key}: no added arguments/markup`);
  }
  for (const token of ['true', 'false']) {
    assert(new RegExp(`\\b${token}\\b`).test(target['wire.settings.account_policy.legacy.help']), token);
  }
  assert(/\bvalue\b/.test(target['wire.settings.account_policy.error.invalid_shape']));
  assert(/\bJSON\b/.test(target['wire.settings.account_policy.error.invalid_json']));
}
validate(fr);
console.log('PASS: valid JSON; 14/14 exact keys in English source order; nonempty French values; zero ICU arguments in source and target; no bidi controls; true/false/value/JSON preserved.');
let negatives = 0;
const missing = { ...fr }; delete missing['wire.settings.account_policy.label'];
assert.throws(() => validate(missing)); negatives++;
assert.throws(() => validate({ ...fr, 'wire.settings.account_policy.help': '{unexpected}' })); negatives++;
assert.throws(() => validate({ ...fr, 'wire.settings.account_policy.error.invalid_shape': 'Seul le champ valeur est attendu.' })); negatives++;
console.log(`PASS: ${negatives} negative controls rejected (missing key, unexpected ICU argument, translated value token).`);
console.log('INFO: no plural/select messages exist; CLDR category validation is not applicable.');
console.log(`INFO: ${Object.values(context).filter(meta => meta.maxCells === 29).length} bounded labels require independent cell-width verification.`);
console.log('NOT RUN: full catalogue validator, scanner, native app/build, pseudo-localization, RTL rendering, rendered frames, independent semantic audits.');
