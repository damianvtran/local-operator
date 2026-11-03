import {createRequire} from 'node:module';
import {readFileSync,writeFileSync} from 'node:fs';
import assert from 'node:assert/strict';
const require=createRequire('/Users/damian/local-operator-ui/package.json');
const {parse,TYPE}=require('@formatjs/icu-messageformat-parser');
const root=new URL('./',import.meta.url);
const en=JSON.parse(readFileSync(new URL('en.json',root)));
const context=JSON.parse(readFileSync(new URL('context.json',root)));
const target=JSON.parse(readFileSync(new URL('wire.errors.sdk_spec.json',root)));
const prefix='wire.errors.sdk_spec.';
const tokens={
agent_identity_conflict:['agent_name','agent_id'],
approval_callback_gate_required:['ApprovalPolicy.callback()'],
approval_declared_tools_empty:['ApprovalPolicy.declared()','ApprovalPolicy.refuse()'],
approval_handler_wrong_mode:['ApprovalPolicy.handler',"''callback''"],
approval_tools_wrong_mode:['ApprovalPolicy.tools',"''declared''"],
attach_fields_unsupported:["mode=''attach''",'spec.resume','spec','resume',"mode=''own''"],
attach_output_unsupported:[],
attach_resume_required:["mode=''attach''",'spec.resume'],
mode_invalid:['mode',"''own''","''attach''"],
output_format_invalid:['output_format'],
output_retries_range:['output_retries','0','5'],
output_retries_requires_format:['output_retries','output_format'],
output_schema_requires_format:['output_schema','output_format'],
resume_empty:['resume','"@latest"'],
spawn_birth_effort_pair_required:['birth_effort','hosting/model','open_session'],
spawn_fields_unsupported:['spawn_session()','spec','open_session','dispose','resume'],
spawn_model_pair_required:['spawn_session()','hosting/model'],
spawn_output_unsupported:['spawn_session()',"open_session(..., mode=''own'')"],
tools_empty:['tools','None'],
world_root_empty:['world_root'],
yolo_approval_unreachable:[en[prefix+'yolo_approval_unreachable'].split(' makes ')[0],'ApprovalPolicy.{approval_mode}()','ApprovalPolicy.auto()','yolo']
};
function shape(message){return parse(message).filter(e=>e.type!==TYPE.literal).map(e=>({type:e.type,value:e.value}));}
function validate(data){
 assert.deepEqual(Object.keys(data),Object.keys(en),'exact keys and order');
 assert.equal(Object.keys(data).length,21);
 for(const [k,v] of Object.entries(data)){
  assert.equal(typeof v,'string');assert(v!==en[k],'not English duplication');assert(/[\u3400-\u9fff]/u.test(v),'Chinese prose');
  assert.deepEqual(shape(v),shape(en[k]),'exact ICU argument names/types/order: '+k);
  assert(parse(v).every(e=>[TYPE.literal,TYPE.argument].includes(e.type)),'no added ICU branches');
  assert(!/[\u061c\u200e\u200f\u202a-\u202e\u2066-\u2069]/u.test(v),'no manual bidi');
  assert(!v.includes('请'),'terse style: source has no please');
  for(const token of tokens[k.slice(prefix.length)])assert(v.includes(token),'preserved token '+token);
  assert(!context[k].maxCells,'no width cap in input');
 }
}
const log=[];function emit(s){log.push(s);console.log(s);}
validate(target);emit('PASS: 21/21 JSON keys; exact English key order; actual Chinese translations; ICU argument names/types/order; all technical token checks; no bidi; source has no width caps');
const negatives=[
 ['missing-key',d=>{delete d[prefix+'tools_empty'];}],
 ['renamed-placeholder',d=>{d[prefix+'mode_invalid']=d[prefix+'mode_invalid'].replace('{mode}','{wrong_mode}');}],
 ['wrong-placeholder-type',d=>{d[prefix+'mode_invalid']=d[prefix+'mode_invalid'].replace('{mode}','{mode, number}');}],
 ['mutated-identifier',d=>{d[prefix+'tools_empty']=d[prefix+'tools_empty'].replace('None','none');}],
 ['English-duplication',d=>{d[prefix+'tools_empty']=en[prefix+'tools_empty'];}],
 ['bidi-injection',d=>{d[prefix+'tools_empty']+='\u2067';}],
 ['malformed-ICU',d=>{d[prefix+'output_format_invalid']+='{';}],
 ['quote-token-loss',d=>{d[prefix+'attach_resume_required']=d[prefix+'attach_resume_required'].replace("mode=''attach''","mode=attach");}]
];
for(const[name,mutate]of negatives){const d=structuredClone(target);mutate(d);let rejected=false;try{validate(d);}catch{rejected=true;}assert(rejected,'negative control must fail: '+name);emit('PASS negative control rejected: '+name);}
const opaque="'unknown' {not_icu} <tool>";
for(const[k,v]of Object.entries(target)){for(const message of [v,en[k]]){const rendered=parse(message).map(e=>e.type===TYPE.literal?e.value:opaque).join('');for(const e of parse(message).filter(e=>e.type===TYPE.argument))assert(rendered.includes(opaque));}}
emit('PASS: parser-AST interpolation smoke preserves opaque argument data without reparsing (not production renderer proof)');
const result={status:'PASS',key_count:21,parser:'existing @formatjs/icu-messageformat-parser via UI dependencies',heap_limit_MiB:256,negative_controls:negatives.map(([name])=>({name,rejected:true})),rendered_fit:'NOT RUN',pseudo:'NOT RUN',RTL:'NOT RUN',production_renderer:'NOT RUN',log};
writeFileSync(new URL('mechanical-validation.json',root),JSON.stringify(result,null,2)+'\n');
writeFileSync(new URL('mechanical.log',root),log.join('\n')+'\n');
