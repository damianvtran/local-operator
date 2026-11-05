import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
const require=createRequire(`${process.env.HOME}/local-operator-ui/package.json`);
const {parse,TYPE}=require('@formatjs/icu-messageformat-parser');
const root=path.dirname(fileURLToPath(import.meta.url));
const namespace=process.argv[2];
const load=f=>JSON.parse(readFileSync(path.join(root,f),'utf8'));
const en=load('inputs/en.json'),fr=load(`${namespace}.json`);
function validate(target){
 assert.deepEqual(Object.keys(target),Object.keys(en),'exact key parity/order');
 for(const [key,text]of Object.entries(target)){
  assert.equal(typeof text,'string'); assert(text.trim()); assert.notEqual(text,en[key]);
  assert(!/[\u061c\u200e\u200f\u202a-\u202e\u2066-\u2069]/u.test(text),'no manual bidi');
  assert(parse(en[key]).every(n=>n.type===TYPE.literal),'argument-free source');
  assert(parse(text).every(n=>n.type===TYPE.literal),'no added ICU arguments/tags');
  assert.deepEqual(text.match(/\d+/g)||[],en[key].match(/\d+/g)||[],'numeric literals retained');
 }
}
validate(fr);
console.log(`PASS ${namespace}: ${Object.keys(fr).length}/${Object.keys(en).length} keys; JSON; source key order; literal-only ICU; no bidi; numeric literals preserved.`);
const missing={...fr}; delete missing[Object.keys(fr)[0]]; assert.throws(()=>validate(missing));
const unexpected={...fr,[Object.keys(fr)[0]]:'{unexpected}'}; assert.throws(()=>validate(unexpected));
console.log('PASS: negative controls rejected missing key and unexpected argument.');
console.log('CLDR: not applicable, zero plural/select arguments in source and target.');
console.log('NOT RUN: full validator/scanner/build/native/rendered/pseudo/RTL/independent semantic audits.');
